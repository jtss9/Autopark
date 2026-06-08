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

from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff, path_length as _path_length
from rl_env import ParkingEnv, CURRICULUM_STAGES
from trajectory import TrajectoryResult, Waypoint

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LOG_DIR = Path(__file__).parent.parent / "logs"

ALGO_REGISTRY = {}


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env: ParkingEnv, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self._last_check = 0
        self._eval_episodes = 50

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


def _make_envs(
    pc: ParkingConfig, cc: CarConfig, stage: int, seed: int,
    goal_conditioned: bool = False,
):
    env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=stage, seed=seed,
        goal_conditioned=goal_conditioned,
    )
    eval_env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=stage, seed=seed + 1000,
        goal_conditioned=goal_conditioned,
    )
    return env, eval_env


def _make_callbacks(
    eval_env, algo_name: str, curriculum: bool, verbose: int,
):
    callbacks = []
    if curriculum:
        callbacks.append(CurriculumCallback(eval_env, check_freq=10000, verbose=verbose))

    save_dir = str(CHECKPOINT_DIR / algo_name / "best")
    os.makedirs(save_dir, exist_ok=True)

    eval_cb = EvalCallback(
        Monitor(eval_env),
        best_model_save_path=save_dir,
        log_path=str(LOG_DIR / algo_name / "eval"),
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
    )
    callbacks.append(eval_cb)
    return callbacks


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------

def train_sac(
    pc: ParkingConfig, cc: CarConfig,
    total_timesteps: int = 1_000_000,
    curriculum: bool = True, start_stage: int = 0,
    seed: int = 0, verbose: int = 1, device: str = "auto",
) -> SAC:
    env, eval_env = _make_envs(pc, cc, start_stage, seed)
    env = Monitor(env)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(LOG_DIR / "sac" / "tensorboard"),
    )

    callbacks = _make_callbacks(eval_env, "sac", curriculum, verbose)

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

    callbacks = _make_callbacks(eval_env, "td3", curriculum, verbose)

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

    callbacks = _make_callbacks(eval_env, "ppo", curriculum, verbose)

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

    callbacks = _make_callbacks(eval_env, "sac_her", curriculum, verbose)

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
    max_steps: int = 300,
) -> Dict[str, Any]:
    goal_conditioned = (algo_name == "sac_her")
    env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=0,
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
