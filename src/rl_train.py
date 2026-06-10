"""
Multi-algorithm RL training for parking: SAC, TD3, PPO, SAC+HER.

Usage:
  python src/rl_train.py --algo sac --timesteps 1000000 --device cuda
  python src/rl_train.py --algo td3 --timesteps 1000000
  python src/rl_train.py --algo ppo --timesteps 1000000
  python src/rl_train.py --algo sac_her --timesteps 1000000
  python src/rl_train.py --algo all --timesteps 1000000   # train all & compare
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from stable_baselines3 import SAC, TD3, PPO, HerReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
except ImportError as e:
    raise ImportError(
        "RL training requires stable-baselines3 and torch. "
        "Install with: pip install -r requirements-rl.txt"
    ) from e

import torch

# Small MLPs + many-core servers: unbounded OpenMP threads thrash. Four
# threads measured ~2.4x faster than the default on this workload.
torch.set_num_threads(4)

from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff, path_length as _path_length, split_by_gear
from rl_env import ParkingEnv, CURRICULUM_STAGES
from trajectory import TrajectoryResult, Waypoint

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LOG_DIR = Path(__file__).parent.parent / "logs"

ALGO_REGISTRY = {}


# ---------------------------------------------------------------------------
# Demonstration seeding: replay expert trajectories into the replay buffer
# ---------------------------------------------------------------------------

def _pure_pursuit_step(env: ParkingEnv, seg: List[Waypoint], gear: int,
                       wp_idx: int, lookahead: float = 0.6,
                       ) -> Tuple[np.ndarray, int]:
    """One pure-pursuit action tracking ``seg`` in the given gear (+1/-1).

    Mirrors tracker.py: find the closest waypoint (monotonically advancing
    ``wp_idx``), then walk ``lookahead`` metres of arc length ahead to pick
    the target. Short lookahead avoids cutting inside min-radius arcs.
    """
    cc = env.cc
    cx, cy, ct = env._car.x, env._car.y, env._car.theta

    best_i, best_d = wp_idx, float("inf")
    for i in range(wp_idx, min(len(seg), wp_idx + 40)):
        d = (seg[i].x - cx) ** 2 + (seg[i].y - cy) ** 2
        if d < best_d:
            best_d, best_i = d, i
    wp_idx = best_i

    target_i = wp_idx
    accum = 0.0
    while target_i < len(seg) - 1 and accum < lookahead:
        a, b = seg[target_i], seg[target_i + 1]
        accum += math.hypot(b.x - a.x, b.y - a.y)
        target_i += 1
    target = seg[target_i]

    dx, dy = target.x - cx, target.y - cy
    cos_t, sin_t = math.cos(ct), math.sin(ct)
    lx = cos_t * dx + sin_t * dy
    ly = -sin_t * dx + cos_t * dy
    if gear < 0:
        # Reversing: a virtual car at heading theta+pi driving forward traces
        # the same path with steering -delta. Mirror the target into the
        # virtual frame, run pure pursuit there, then negate the steering.
        lx, ly = -lx, -ly

    d2 = lx * lx + ly * ly
    steer_angle = math.atan(2.0 * ly / d2 * cc.wheelbase) if d2 > 1e-6 else 0.0
    if gear < 0:
        steer_angle = -steer_angle
    steer_cmd = float(np.clip(steer_angle / cc.max_steer, -1.0, 1.0))

    end = seg[-1]
    remaining = math.hypot(end.x - cx, end.y - cy)
    max_ms = 1.0 if gear > 0 else 0.6
    speed_ms = min(0.1 + 0.45 * remaining, max_ms)  # slow near segment end
    speed_cmd = float(np.clip(gear * speed_ms / env.max_speed, -1.0, 1.0))
    return np.array([steer_cmd, speed_cmd], dtype=np.float32), wp_idx


def collect_demonstrations(
    pc: ParkingConfig, cc: CarConfig,
    n_demos: int = 30,
    perturb_pos: float = 0.5,
    perturb_heading: float = 0.14,  # ~8 deg
) -> List[Dict[str, Any]]:
    """Track the hybrid A* reference path through the env with a gear-aware
    pure-pursuit controller, recording real (obs, action, reward, next_obs,
    done) transitions from the fixed start pose (small perturbations).

    Unlike the previous version this drives each contiguous episode through
    the env (no per-segment resets) and actually reverses on reverse gears.
    """
    from trajectory import _densify_for_tracking

    env = ParkingEnv(parking_config=pc, car_config=cc, max_episode_steps=600)
    if env._ref_path is None:
        print("  [Demos] No hybrid A* reference path — skipping demo seeding")
        return []

    ref_wps = [Waypoint(x, y, th) for x, y, th in env._ref_path]
    ref_wps = _densify_for_tracking(ref_wps, max_step=0.15)
    segments = split_by_gear(ref_wps)

    rng = np.random.default_rng(42)
    transitions: List[Dict[str, Any]] = []
    successes = 0
    fx, fy, ftheta = env.lot.car_start_pose

    for demo_i in range(n_demos):
        if demo_i == 0:
            start = (fx, fy, ftheta)
        else:
            start = (
                fx + rng.uniform(-perturb_pos, perturb_pos),
                fy + rng.uniform(-perturb_pos * 0.5, perturb_pos * 0.5),
                ftheta + rng.uniform(-perturb_heading, perturb_heading),
            )
            if not env.grid.pose_is_valid(start):
                start = (fx, fy, ftheta)

        obs, info = env.reset(options={"start_pose": start})
        ep_transitions: List[Dict[str, Any]] = []
        done = False

        for seg_i, (gear, seg) in enumerate(segments):
            if done:
                break
            end = seg[-1]
            wp_idx = 0
            for _ in range(300):
                remaining = math.hypot(end.x - env._car.x, end.y - env._car.y)
                if remaining < 0.12:
                    break
                action, wp_idx = _pure_pursuit_step(env, seg, gear, wp_idx)
                next_obs, reward, terminated, truncated, info = env.step(action)
                ep_transitions.append({
                    "obs": obs.copy(),
                    "action": action.copy(),
                    "reward": reward,
                    "next_obs": next_obs.copy(),
                    "done": terminated,
                })
                obs = next_obs
                if terminated or truncated:
                    done = True
                    break

        if info.get("is_success", False):
            successes += 1
        transitions.extend(ep_transitions)

    print(f"  [Demos] Collected {len(transitions)} transitions from "
          f"{n_demos} demos ({successes} successful)")
    return transitions


def seed_replay_buffer(model, transitions: List[Dict[str, Any]]):
    """Insert demonstration transitions into the model's replay buffer."""
    buf = model.replay_buffer
    for t in transitions:
        buf.add(
            obs=t["obs"].reshape(1, -1),
            next_obs=t["next_obs"].reshape(1, -1),
            action=t["action"].reshape(1, -1),
            reward=np.array([t["reward"]]),
            done=np.array([t["done"]]),
            infos=[{}],
        )


def bc_pretrain_actor(
    model: SAC, transitions: List[Dict[str, Any]],
    epochs: int = 15, batch_size: int = 512, lr: float = 1e-3,
):
    """Behaviour-clone the demo actions into SAC's actor before RL.

    Supervises tanh(mu(latent_pi(obs))) — the deterministic action — with
    MSE against the demo actions, so RL starts from a policy that can
    already roughly execute the maneuver instead of discovering it from
    scratch through Q-learning alone.
    """
    actor = model.policy.actor
    device = model.device
    obs_t = torch.as_tensor(
        np.stack([t["obs"] for t in transitions]), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(
        np.stack([t["action"] for t in transitions]), dtype=torch.float32, device=device)

    params = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    n = len(obs_t)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            features = actor.extract_features(obs_t[idx], actor.features_extractor)
            mean = actor.mu(actor.latent_pi(features))
            loss = torch.nn.functional.mse_loss(torch.tanh(mean), act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if epoch == 0 or epoch == epochs - 1:
            print(f"  [BC-init] epoch {epoch + 1}/{epochs}: loss={total / n:.5f}")


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DemoRegCallback(BaseCallback):
    """SACfD-style demonstration regularization.

    Each env step also takes one small BC gradient step pulling the actor's
    deterministic action toward the demo actions, with a weight that decays
    to zero over ``decay_steps``. This stops early Q-learning noise from
    washing out the demonstrated maneuver while still letting RL own the
    final policy.
    """

    def __init__(self, demos: List[Dict[str, Any]], lr: float = 1e-4,
                 batch_size: int = 256, decay_steps: int = 400_000,
                 verbose: int = 0):
        super().__init__(verbose)
        self._demos = demos
        self.lr = lr
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        self._obs = None

    def _on_training_start(self) -> None:
        device = self.model.device
        self._obs = torch.as_tensor(
            np.stack([t["obs"] for t in self._demos]),
            dtype=torch.float32, device=device)
        self._act = torch.as_tensor(
            np.stack([t["action"] for t in self._demos]),
            dtype=torch.float32, device=device)
        actor = self.model.policy.actor
        params = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
        self._optimizer = torch.optim.Adam(params, lr=self.lr)

    def _on_step(self) -> bool:
        if self.num_timesteps < self.model.learning_starts:
            return True
        weight = max(0.0, 1.0 - self.num_timesteps / self.decay_steps)
        if weight <= 0.0:
            return True
        idx = torch.randint(0, len(self._obs), (self.batch_size,),
                            device=self._obs.device)
        actor = self.model.policy.actor
        features = actor.extract_features(self._obs[idx], actor.features_extractor)
        mean = actor.mu(actor.latent_pi(features))
        loss = weight * torch.nn.functional.mse_loss(
            torch.tanh(mean), self._act[idx])
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return True


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env: ParkingEnv, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self._last_check = 0
        self._eval_episodes = 30

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps

        train_env = self.training_env.envs[0]
        if hasattr(train_env, "env"):
            train_env = train_env.env

        stage_idx = train_env.curriculum_stage
        if stage_idx >= len(CURRICULUM_STAGES) - 1:
            return True

        success_count = 0
        for _ in range(self._eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            if info.get("is_success", False):
                success_count += 1

        rate = success_count / self._eval_episodes
        threshold = CURRICULUM_STAGES[stage_idx].success_rate_threshold

        if self.verbose:
            print(
                f"  [Curriculum] stage={stage_idx} "
                f"success={rate:.0%} threshold={threshold:.0%}"
            )

        if rate >= threshold:
            new_stage = stage_idx + 1
            train_env.curriculum_stage = new_stage
            self.eval_env.curriculum_stage = new_stage
            if self.verbose:
                print(
                    f"  [Curriculum] PROMOTED to stage {new_stage}: "
                    f"{CURRICULUM_STAGES[new_stage].name}"
                )
        return True


class FixedStartEvalCallback(BaseCallback):
    """Evaluate from the lot's fixed start pose (the `python main.py`
    scenario) and checkpoint whenever that success rate improves.

    EvalCallback selects by mean reward over the curriculum's random starts,
    which is not the metric we ship; this one is.
    """

    def __init__(
        self, pc: ParkingConfig, cc: CarConfig, save_path: str,
        eval_freq: int = 10_000, n_episodes: int = 10, verbose: int = 1,
        env=None,
    ):
        super().__init__(verbose)
        self.env = env if env is not None else ParkingEnv(
            parking_config=pc, car_config=cc, max_episode_steps=250, seed=123,
        )
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self._last_eval = 0
        self.best_rate = -1.0
        self.best_reward = -np.inf
        self._rng = np.random.default_rng(7)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        fx, fy, ftheta = self.env.lot.car_start_pose
        successes = 0
        total_reward = 0.0
        for ep in range(self.n_episodes):
            if ep == 0:
                start = (fx, fy, ftheta)
            else:
                start = (
                    fx + self._rng.uniform(-0.5, 0.5),
                    fy + self._rng.uniform(-0.25, 0.25),
                    ftheta + self._rng.uniform(-0.14, 0.14),
                )
                if not self.env.grid.pose_is_valid(start):
                    start = (fx, fy, ftheta)
            obs, info = self.env.reset(options={"start_pose": start})
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            if info.get("is_success", False):
                successes += 1

        rate = successes / self.n_episodes
        mean_reward = total_reward / self.n_episodes
        if self.verbose:
            print(f"  [FixedStart] t={self.num_timesteps} "
                  f"success={rate:.0%} mean_reward={mean_reward:.1f}")

        if rate > self.best_rate or (
            rate == self.best_rate and mean_reward > self.best_reward
        ):
            self.best_rate = rate
            self.best_reward = mean_reward
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
            if self.verbose:
                print(f"  [FixedStart] new best ({rate:.0%}) → {self.save_path}")
        return True


def _make_envs(
    pc: ParkingConfig, cc: CarConfig, stage: int, seed: int,
    goal_conditioned: bool = False,
):
    env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=stage, seed=seed,
        goal_conditioned=goal_conditioned,
        max_episode_steps=200,
    )
    eval_env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=stage, seed=seed + 1000,
        goal_conditioned=goal_conditioned,
        max_episode_steps=250,
    )
    return env, eval_env


def _make_callbacks(
    eval_env, algo_name: str, curriculum: bool, verbose: int,
    pc: Optional[ParkingConfig] = None, cc: Optional[CarConfig] = None,
):
    callbacks = []
    if curriculum:
        callbacks.append(CurriculumCallback(eval_env, check_freq=15000, verbose=verbose))

    save_dir = str(CHECKPOINT_DIR / algo_name / "best")
    os.makedirs(save_dir, exist_ok=True)

    eval_cb = EvalCallback(
        Monitor(eval_env),
        best_model_save_path=save_dir,
        log_path=str(LOG_DIR / algo_name / "eval"),
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
    )
    callbacks.append(eval_cb)

    if pc is not None and cc is not None and not eval_env.goal_conditioned:
        callbacks.append(FixedStartEvalCallback(
            pc, cc,
            save_path=str(CHECKPOINT_DIR / algo_name / "best_fixed" / "model"),
            eval_freq=15_000, n_episodes=10, verbose=verbose,
        ))
    return callbacks


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------

def train_sac(
    pc: ParkingConfig, cc: CarConfig,
    total_timesteps: int = 1_000_000,
    curriculum: bool = True, start_stage: int = 0,
    seed: int = 0, verbose: int = 1, device: str = "auto",
    demo_seeding: bool = True, n_demos: int = 60,
    demo_reg: bool = True,
) -> SAC:
    env, eval_env = _make_envs(pc, cc, start_stage, seed)
    env = Monitor(env)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=2_000 if demo_seeding else 5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        # Precision parking needs tight actions near the goal: the default
        # target entropy (-2) keeps sigma~0.37 per action dim, which floods
        # the buffer with collisions in the tight spot-entry passage.
        target_entropy=-4.0,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(LOG_DIR / "sac" / "tensorboard"),
    )

    demos = []
    if demo_seeding:
        print(f"[SAC] Seeding replay buffer with {n_demos} hybrid A* demos...")
        demos = collect_demonstrations(pc, cc, n_demos=n_demos)
        if demos:
            seed_replay_buffer(model, demos)
            print(f"[SAC] Seeded {len(demos)} transitions")
            bc_pretrain_actor(model, demos)

    callbacks = _make_callbacks(eval_env, "sac", curriculum, verbose, pc, cc)
    if demo_reg and demos:
        callbacks.append(DemoRegCallback(demos, decay_steps=400_000))
        print("[SAC] Demo regularization enabled (decay over 400k steps)")

    save_path = str(CHECKPOINT_DIR / "sac" / "final")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[SAC] Training for {total_timesteps} steps on {device}...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(save_path)
    print(f"[SAC] Saved to {save_path}")
    return model


ALGO_REGISTRY["sac"] = train_sac


# ---------------------------------------------------------------------------
# TD3
# ---------------------------------------------------------------------------

def train_td3(
    pc: ParkingConfig, cc: CarConfig,
    total_timesteps: int = 1_000_000,
    curriculum: bool = True, start_stage: int = 0,
    seed: int = 0, verbose: int = 1, device: str = "auto",
) -> TD3:
    env, eval_env = _make_envs(pc, cc, start_stage, seed)
    env = Monitor(env)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        train_freq=1,
        gradient_steps=1,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(LOG_DIR / "td3" / "tensorboard"),
    )

    callbacks = _make_callbacks(eval_env, "td3", curriculum, verbose, pc, cc)

    save_path = str(CHECKPOINT_DIR / "td3" / "final")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[TD3] Training for {total_timesteps} steps on {device}...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(save_path)
    print(f"[TD3] Saved to {save_path}")
    return model


ALGO_REGISTRY["td3"] = train_td3


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def train_ppo(
    pc: ParkingConfig, cc: CarConfig,
    total_timesteps: int = 1_000_000,
    curriculum: bool = True, start_stage: int = 0,
    seed: int = 0, verbose: int = 1, device: str = "auto",
) -> PPO:
    env, eval_env = _make_envs(pc, cc, start_stage, seed)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(LOG_DIR / "ppo" / "tensorboard"),
    )

    callbacks = _make_callbacks(eval_env, "ppo", curriculum, verbose, pc, cc)

    save_path = str(CHECKPOINT_DIR / "ppo" / "final")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[PPO] Training for {total_timesteps} steps on {device}...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(save_path)
    print(f"[PPO] Saved to {save_path}")
    return model


ALGO_REGISTRY["ppo"] = train_ppo


# ---------------------------------------------------------------------------
# SAC + HER (Hindsight Experience Replay) — highway-env standard approach
# ---------------------------------------------------------------------------

def train_sac_her(
    pc: ParkingConfig, cc: CarConfig,
    total_timesteps: int = 1_000_000,
    curriculum: bool = True, start_stage: int = 0,
    seed: int = 0, verbose: int = 1, device: str = "auto",
) -> SAC:
    env, eval_env = _make_envs(
        pc, cc, start_stage, seed, goal_conditioned=True
    )
    env = Monitor(env)

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.05,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(LOG_DIR / "sac_her" / "tensorboard"),
    )

    callbacks = _make_callbacks(eval_env, "sac_her", curriculum, verbose, pc, cc)

    save_path = str(CHECKPOINT_DIR / "sac_her" / "final")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[SAC+HER] Training for {total_timesteps} steps on {device}...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(save_path)
    print(f"[SAC+HER] Saved to {save_path}")
    return model


ALGO_REGISTRY["sac_her"] = train_sac_her


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model, algo_name: str,
    pc: ParkingConfig, cc: CarConfig,
    n_episodes: int = 50,
    max_steps: int = 500,
) -> Dict[str, Any]:
    goal_conditioned = (algo_name == "sac_her")
    # Evaluate at the hardest no-obstacle stage so "random" starts cover the
    # full maneuver, not just the easy near-goal starts of stage 0.
    eval_stage = max(
        i for i, s in enumerate(CURRICULUM_STAGES)
        if s.obstacle_scenario == "none"
    )
    env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=eval_stage,
        max_episode_steps=max_steps,
        goal_conditioned=goal_conditioned,
    )

    results = {
        "algo": algo_name,
        "episodes": n_episodes,
        "successes": 0,
        "collisions": 0,
        "timeouts": 0,
        "avg_reward": 0.0,
        "avg_steps": 0.0,
        "avg_pos_error": 0.0,
        "avg_heading_error": 0.0,
        "from_fixed_start": {"successes": 0, "total": 0},
        "from_random_start": {"successes": 0, "total": 0},
    }

    total_reward = 0.0
    total_steps = 0
    total_pos_err = 0.0
    total_heading_err = 0.0

    for ep in range(n_episodes):
        use_fixed = (ep % 2 == 0)
        if use_fixed:
            obs, info = env.reset(options={"start_pose": env.lot.car_start_pose})
            results["from_fixed_start"]["total"] += 1
        else:
            obs, info = env.reset()
            results["from_random_start"]["total"] += 1

        ep_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        total_reward += ep_reward
        total_steps += steps
        total_pos_err += info.get("pos_error", 0)
        total_heading_err += info.get("heading_error_deg", 0)

        if info.get("is_success", False):
            results["successes"] += 1
            if use_fixed:
                results["from_fixed_start"]["successes"] += 1
            else:
                results["from_random_start"]["successes"] += 1
        elif info.get("collision", False):
            results["collisions"] += 1
        else:
            results["timeouts"] += 1

    results["avg_reward"] = total_reward / n_episodes
    results["avg_steps"] = total_steps / n_episodes
    results["avg_pos_error"] = total_pos_err / n_episodes
    results["avg_heading_error"] = total_heading_err / n_episodes
    results["success_rate"] = results["successes"] / n_episodes
    results["collision_rate"] = results["collisions"] / n_episodes

    fs = results["from_fixed_start"]
    rs = results["from_random_start"]
    fs["success_rate"] = fs["successes"] / max(fs["total"], 1)
    rs["success_rate"] = rs["successes"] / max(rs["total"], 1)

    return results


def print_comparison(all_results: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print(f"{'Algorithm':<12} {'Success%':>8} {'Collision%':>10} "
          f"{'AvgReward':>10} {'AvgSteps':>8} "
          f"{'Fixed%':>8} {'Random%':>8} "
          f"{'PosErr':>8} {'HeadErr':>8}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['algo']:<12} "
            f"{r['success_rate']:>7.1%} "
            f"{r['collision_rate']:>9.1%} "
            f"{r['avg_reward']:>10.1f} "
            f"{r['avg_steps']:>8.1f} "
            f"{r['from_fixed_start']['success_rate']:>7.1%} "
            f"{r['from_random_start']['success_rate']:>7.1%} "
            f"{r['avg_pos_error']:>8.3f} "
            f"{r['avg_heading_error']:>8.1f}"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & compare RL parking agents")
    parser.add_argument("--algo", default="sac",
                        choices=["sac", "td3", "ppo", "sac_her", "all"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--parking-type", default="perpendicular",
                        choices=["perpendicular", "parallel"])
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = _default_device()

    pc = ParkingConfig(parking_type=args.parking_type)
    cc = CarConfig()

    algos = list(ALGO_REGISTRY.keys()) if args.algo == "all" else [args.algo]

    all_results = []
    for algo_name in algos:
        print(f"\n{'='*60}")
        print(f"  {algo_name.upper()} — {args.parking_type} parking")
        print(f"{'='*60}")

        train_fn = ALGO_REGISTRY[algo_name]

        if args.eval_only:
            # Try to load existing checkpoint
            ckpt = str(CHECKPOINT_DIR / algo_name / "best" / "best_model.zip")
            if not os.path.exists(ckpt):
                ckpt = str(CHECKPOINT_DIR / algo_name / "final.zip")
            if not os.path.exists(ckpt):
                print(f"  [SKIP] No checkpoint found for {algo_name}")
                continue
            print(f"  Loading {ckpt}...")
            algo_cls = {"sac": SAC, "td3": TD3, "ppo": PPO, "sac_her": SAC}[algo_name]
            model = algo_cls.load(ckpt, device=args.device)
        else:
            t0 = time.perf_counter()
            model = train_fn(
                pc=pc, cc=cc,
                total_timesteps=args.timesteps,
                curriculum=not args.no_curriculum,
                start_stage=args.stage,
                seed=args.seed,
                device=args.device,
            )
            elapsed = time.perf_counter() - t0
            print(f"  Training time: {elapsed:.0f}s")

        print(f"  Evaluating {algo_name} ({args.eval_episodes} episodes)...")
        result = evaluate_model(model, algo_name, pc, cc, n_episodes=args.eval_episodes)
        all_results.append(result)

        print(f"  Success: {result['success_rate']:.1%} "
              f"(fixed: {result['from_fixed_start']['success_rate']:.1%}, "
              f"random: {result['from_random_start']['success_rate']:.1%})")
        print(f"  Collision: {result['collision_rate']:.1%}")
        print(f"  Avg reward: {result['avg_reward']:.1f}")

    if len(all_results) > 1:
        print_comparison(all_results)

    # Save results
    out_path = CHECKPOINT_DIR / "comparison_results.json"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
