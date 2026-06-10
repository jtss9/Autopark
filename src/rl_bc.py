"""
Behavioral Cloning → RL Fine-tuning for parking.

1. Collect dense (obs, action) demonstrations by tracking hybrid A* waypoints
   with a PD controller through the actual env.
2. Train a BC policy (supervised MLP) to imitate the demonstrations.
3. Load the BC weights into SAC and fine-tune with RL.

Usage:
  python src/rl_bc.py --timesteps 1000000 --device cuda
  python src/rl_bc.py --bc-only          # just train BC, no RL
  python src/rl_bc.py --eval-only        # evaluate existing checkpoint
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as e:
    raise ImportError("pip install -r requirements-rl.txt") from e

from config import CarConfig, ParkingConfig
from controller import CarDynamics
from geom import angle_diff, wrap_pi
from rl_env import ParkingEnv, CURRICULUM_STAGES
from rl_train import CurriculumCallback, _make_callbacks
from trajectory import plan_trajectory, Waypoint

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LOG_DIR = Path(__file__).parent.parent / "logs"


class ForcePromotionCallback(BaseCallback):
    """Force curriculum promotion at fixed timestep intervals.

    The standard CurriculumCallback only promotes when the eval success rate
    exceeds a threshold, but if the model only learns to park from close
    starts it never gets enough far-start successes to promote. This callback
    forces promotion after a fixed number of timesteps, ensuring the model
    is eventually exposed to far starts.
    """

    def __init__(
        self, eval_env: ParkingEnv,
        promote_every: int = 300_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.promote_every = promote_every
        self._last_promote = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_promote < self.promote_every:
            return True

        train_env = self.training_env.envs[0]
        if hasattr(train_env, "env"):
            train_env = train_env.env

        stage_idx = train_env.curriculum_stage
        if stage_idx >= len(CURRICULUM_STAGES) - 1:
            return True

        new_stage = stage_idx + 1
        train_env.curriculum_stage = new_stage
        self.eval_env.curriculum_stage = new_stage
        self._last_promote = self.num_timesteps

        if self.verbose:
            print(
                f"  [ForcePromotion] {self.num_timesteps} steps → "
                f"stage {new_stage}: {CURRICULUM_STAGES[new_stage].name} "
                f"(dist={CURRICULUM_STAGES[new_stage].max_start_distance}m)"
            )
        return True


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 1. Dense demonstration collection via PD waypoint tracking
# ---------------------------------------------------------------------------

def _pure_pursuit_action(
    car_x: float, car_y: float, car_theta: float,
    waypoints: List[Waypoint], wp_idx: int,
    cc: CarConfig, max_speed: float,
    reverse: bool = False,
    lookahead: float = 1.5,
) -> Tuple[float, float, int]:
    """Pure-pursuit controller that returns (steer_cmd, speed_cmd, new_wp_idx).

    Looks ahead along the waypoint list to find a target point at ``lookahead``
    distance, then computes the bicycle-model steering angle to reach it.
    """
    # Advance wp_idx past waypoints we've already passed
    while wp_idx < len(waypoints) - 1:
        d = math.hypot(waypoints[wp_idx].x - car_x, waypoints[wp_idx].y - car_y)
        if d > 0.8:
            break
        wp_idx += 1

    # Find lookahead target along the path
    target = waypoints[min(wp_idx, len(waypoints) - 1)]
    for i in range(wp_idx, len(waypoints)):
        d = math.hypot(waypoints[i].x - car_x, waypoints[i].y - car_y)
        if d >= lookahead:
            target = waypoints[i]
            break
    else:
        target = waypoints[-1]

    dx = target.x - car_x
    dy = target.y - car_y
    dist = math.hypot(dx, dy)

    # Transform target to car's local frame
    local_x = dx * math.cos(car_theta) + dy * math.sin(car_theta)
    local_y = -dx * math.sin(car_theta) + dy * math.cos(car_theta)

    if reverse:
        # When reversing, flip x-axis (target is "behind" the car)
        # and invert y for correct steering direction
        local_x = -local_x
        local_y = -local_y

    # Pure pursuit: curvature = 2 * local_y / dist^2
    if dist > 0.01 and abs(local_x) > 0.01:
        curvature = 2.0 * local_y / (local_x * local_x + local_y * local_y)
        steer_angle = math.atan(curvature * cc.wheelbase)
    else:
        # When target is very close or directly lateral, steer toward it
        steer_angle = math.copysign(cc.max_steer * 0.5, local_y)

    steer_cmd = np.clip(steer_angle / cc.max_steer, -1.0, 1.0)

    # Slow down when steering hard or near end of path
    steer_factor = 1.0 - 0.6 * abs(steer_cmd)
    remaining = len(waypoints) - 1 - wp_idx
    approach_factor = min(1.0, remaining / 5.0)
    speed_mag = np.clip(dist * 1.0 * steer_factor * approach_factor, 0.15, 0.5)
    speed_cmd = -speed_mag if reverse else speed_mag

    return float(steer_cmd), float(speed_cmd), wp_idx


def _astar_segment_demos(
    env: ParkingEnv, cc: CarConfig,
    wps: List[Waypoint],
) -> Tuple[List[Dict[str, np.ndarray]], bool]:
    """Generate demos by resetting to each A* waypoint and tracking to the next.

    Avoids error accumulation by resetting the env per segment. Each segment
    produces (obs, action) pairs that teach the correct action for that state.
    The final segment does a straight reverse from near-goal to goal.
    """
    transitions = []
    goal = env._goal_pose

    # Compute forward/reverse per segment
    seg_reverse = []
    for i in range(1, len(wps)):
        dx = wps[i].x - wps[i - 1].x
        dy = wps[i].y - wps[i - 1].y
        fwd = dx * math.cos(wps[i - 1].theta) + dy * math.sin(wps[i - 1].theta)
        seg_reverse.append(fwd < -0.01)

    success = False
    for seg_i in range(len(wps) - 1):
        curr = wps[seg_i]
        target = wps[seg_i + 1]
        is_rev = seg_reverse[seg_i]

        dx = target.x - curr.x
        dy = target.y - curr.y
        dist = math.hypot(dx, dy)
        dtheta = angle_diff(target.theta, curr.theta)

        env.reset(options={"start_pose": (curr.x, curr.y, curr.theta)})

        if abs(dtheta) > 0.001 and dist > 0.001:
            R_actual = dist / (2 * abs(math.sin(dtheta / 2)))
            steer_angle = math.atan(cc.wheelbase / R_actual)
            if dtheta < 0:
                steer_angle = -steer_angle
        else:
            steer_angle = 0.0

        steer = float(np.clip(steer_angle / cc.max_steer, -1.0, 1.0))
        n_substeps = max(3, int(dist / (env.max_speed * env.dt * 0.1)))
        speed_per = dist / n_substeps / env.dt / env.max_speed
        speed = float(np.clip(speed_per, 0.05, 0.2))
        if is_rev:
            speed = -speed

        for _ in range(n_substeps):
            obs = env._get_obs_vector()
            action = np.array([steer, speed], dtype=np.float32)
            transitions.append({"obs": obs.copy(), "action": action.copy()})
            _, _, term, trunc, info = env.step(action)
            if info.get("collision"):
                break

    # Final phase: straight reverse from last waypoint to goal
    env.reset(options={"start_pose": (goal[0], wps[-2].y, goal[2])})
    for _ in range(400):
        cx, cy, ct = env._car.x, env._car.y, env._car.theta
        x_err = goal[0] - cx
        heading_err = angle_diff(goal[2], ct)
        steer = float(np.clip(-0.3 * x_err + 0.05 * heading_err, -0.05, 0.05))
        d = math.hypot(cx - goal[0], cy - goal[1])
        speed = float(np.clip(-d * 0.08, -0.1, -0.03))
        obs = env._get_obs_vector()
        action = np.array([steer, speed], dtype=np.float32)
        transitions.append({"obs": obs.copy(), "action": action.copy()})
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            success = info.get("is_success", False)
            break

    return transitions, success


def collect_dense_demos(
    pc: ParkingConfig, cc: CarConfig,
    n_demos: int = 200,
    **kwargs,
) -> List[Dict[str, np.ndarray]]:
    """Collect demonstrations by replaying hybrid A* segments.

    Uses per-segment reset to avoid error accumulation. Each demo
    generates the full set of (obs, action) pairs for the A* solution.
    Multiple demos with slight variations build a robust dataset.
    """
    base_pc = ParkingConfig(
        parking_type=pc.parking_type, planner="hybrid_astar",
        lane_width=pc.lane_width, spot_length=pc.spot_length,
        spot_width=pc.spot_width,
    )
    print("  Planning base A* trajectory...", end=" ", flush=True)
    result = plan_trajectory(base_pc, cc)
    if not result.feasible:
        print("FAILED")
        return []
    wps = result.waypoints
    print(f"{len(wps)} waypoints")

    env = ParkingEnv(parking_config=pc, car_config=cc, max_episode_steps=2000)
    all_transitions = []
    successes = 0

    for demo_i in range(n_demos):
        transitions, success = _astar_segment_demos(env, cc, wps)
        all_transitions.extend(transitions)
        if success:
            successes += 1

        if (demo_i + 1) % 50 == 0:
            print(f"    demo {demo_i + 1}/{n_demos}: "
                  f"{len(all_transitions)} transitions, {successes} successes")

    print(f"  [BC Demos] {len(all_transitions)} transitions from "
          f"{n_demos} demos ({successes} successful)")
    return all_transitions


# ---------------------------------------------------------------------------
# 2. Behavioral Cloning (supervised)
# ---------------------------------------------------------------------------

def train_bc(
    transitions: List[Dict[str, np.ndarray]],
    obs_dim: int,
    act_dim: int,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "auto",
) -> nn.Module:
    if device == "auto":
        device = _default_device()

    obs_arr = np.stack([t["obs"] for t in transitions])
    act_arr = np.stack([t["action"] for t in transitions])

    obs_t = torch.FloatTensor(obs_arr).to(device)
    act_t = torch.FloatTensor(act_arr).to(device)

    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, act_dim),
        nn.Tanh(),
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"  [BC] Training on {len(transitions)} samples, "
          f"{epochs} epochs, device={device}")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            print(f"    epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")

    bc_path = CHECKPOINT_DIR / "bc" / "bc_policy.pt"
    os.makedirs(bc_path.parent, exist_ok=True)
    torch.save(policy.state_dict(), bc_path)
    print(f"  [BC] Saved to {bc_path}")
    return policy


# ---------------------------------------------------------------------------
# 3. Load BC weights into SAC, then RL fine-tune
# ---------------------------------------------------------------------------

def _copy_bc_to_sac(bc_policy: nn.Module, sac_model: SAC):
    """Copy BC weights into SAC's actor network (mu layer)."""
    sac_actor = sac_model.policy.actor

    bc_layers = [m for m in bc_policy if isinstance(m, nn.Linear)]
    sac_latent = sac_actor.latent_pi
    sac_mu = sac_actor.mu

    sac_linears = [m for m in sac_latent if isinstance(m, nn.Linear)]
    sac_linears.append(sac_mu)

    copied = 0
    for bc_l, sac_l in zip(bc_layers, sac_linears):
        if bc_l.weight.shape == sac_l.weight.shape:
            sac_l.weight.data.copy_(bc_l.weight.data)
            sac_l.bias.data.copy_(bc_l.bias.data)
            copied += 1

    print(f"  [BC→SAC] Copied {copied}/{len(bc_layers)} layers")
    return copied > 0


def train_bc_then_rl(
    pc: ParkingConfig, cc: CarConfig,
    n_demos: int = 200,
    bc_epochs: int = 100,
    rl_timesteps: int = 1_000_000,
    seed: int = 0,
    device: str = "auto",
) -> SAC:
    if device == "auto":
        device = _default_device()

    # Step 1: Collect demonstrations
    print("=" * 60)
    print("  Phase 1: Collecting demonstrations")
    print("=" * 60)
    transitions = collect_dense_demos(pc, cc, n_demos=n_demos)
    if not transitions:
        raise RuntimeError("No demonstrations collected")

    # Step 2: Train BC
    print("\n" + "=" * 60)
    print("  Phase 2: Behavioral Cloning")
    print("=" * 60)
    obs_dim = transitions[0]["obs"].shape[0]
    act_dim = transitions[0]["action"].shape[0]
    bc_policy = train_bc(
        transitions, obs_dim, act_dim,
        epochs=bc_epochs, device=device,
    )

    # Step 3: Evaluate BC alone
    print("\n" + "=" * 60)
    print("  Phase 2.5: Evaluating BC policy")
    print("=" * 60)
    _evaluate_bc(bc_policy, pc, cc, device, n_episodes=50)

    # Step 4: Create SAC and copy BC weights
    print("\n" + "=" * 60)
    print("  Phase 3: RL Fine-tuning (SAC)")
    print("=" * 60)
    env = ParkingEnv(parking_config=pc, car_config=cc, seed=seed)
    env = Monitor(env)
    eval_env = ParkingEnv(parking_config=pc, car_config=cc, seed=seed + 1000)

    sac = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=500_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(LOG_DIR / "bc_sac" / "tensorboard"),
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    success = _copy_bc_to_sac(bc_policy, sac)
    if not success:
        print("  [WARNING] BC weight copy failed — training SAC from scratch")

    # Seed replay buffer by replaying the two-phase controller through the env
    # Seed replay buffer with per-segment A* demos (same as BC collection)
    env_tmp = ParkingEnv(parking_config=pc, car_config=cc, max_episode_steps=2000)
    print("  Seeding replay buffer via A* segment replays...")
    demo_buf_count = 0

    base_pc = ParkingConfig(parking_type=pc.parking_type, planner="hybrid_astar")
    astar_result = plan_trajectory(base_pc, cc)
    if astar_result.feasible:
        wps = astar_result.waypoints
        for demo_i in range(min(n_demos, 50)):
            seg_transitions, _ = _astar_segment_demos(env_tmp, cc, wps)
            # Re-run each segment's actions through a fresh env for (s,a,r,s')
            for seg_t in seg_transitions:
                # Use the obs/action from the demo directly
                obs_vec = seg_t["obs"]
                action = seg_t["action"]
                # We need next_obs + reward, but can't step sequentially
                # across segment resets. Instead, just add with estimated reward.
                sac.replay_buffer.add(
                    obs=obs_vec.reshape(1, -1),
                    next_obs=obs_vec.reshape(1, -1),  # approximate
                    action=action.reshape(1, -1),
                    reward=np.array([0.1]),  # small positive bias
                    done=np.array([False]),
                    infos=[{}],
                )
                demo_buf_count += 1

    print(f"  Seeded {demo_buf_count} transitions into replay buffer")

    callbacks = _make_callbacks(eval_env, "bc_sac", curriculum=True, verbose=1)
    # Force promotion every 300k steps so model trains on far starts
    callbacks.append(ForcePromotionCallback(eval_env, promote_every=300_000, verbose=1))

    save_path = str(CHECKPOINT_DIR / "bc_sac" / "final")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"  [SAC] Fine-tuning for {rl_timesteps} steps on {device}...")
    print(f"  [SAC] ForcePromotion: stage advances every 300k steps")
    sac.learn(total_timesteps=rl_timesteps, callback=callbacks)
    sac.save(save_path)
    print(f"  [SAC] Saved to {save_path}")
    return sac


def _evaluate_bc(
    bc_policy: nn.Module, pc: ParkingConfig, cc: CarConfig,
    device: str, n_episodes: int = 50,
):
    """Evaluate the BC policy directly (no RL)."""
    env = ParkingEnv(parking_config=pc, car_config=cc, max_episode_steps=500)
    bc_policy.eval()

    successes = 0
    fixed_success = 0
    random_success = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        use_fixed = (ep % 2 == 0)
        if use_fixed:
            obs, _ = env.reset(options={"start_pose": env.lot.car_start_pose})
        else:
            obs, _ = env.reset()

        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = bc_policy(obs_t).cpu().numpy().flatten()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward
        if info.get("is_success", False):
            successes += 1
            if use_fixed:
                fixed_success += 1
            else:
                random_success += 1

    n_fixed = n_episodes // 2
    n_random = n_episodes - n_fixed
    print(f"  [BC Eval] Success={successes}/{n_episodes} "
          f"({successes / n_episodes:.1%}) "
          f"fixed={fixed_success}/{n_fixed} "
          f"random={random_success}/{n_random} "
          f"avg_reward={total_reward / n_episodes:.1f}")


def evaluate_bc_sac(
    pc: ParkingConfig, cc: CarConfig,
    n_episodes: int = 100, device: str = "auto",
):
    """Evaluate the BC+SAC fine-tuned model."""
    if device == "auto":
        device = _default_device()

    candidates = [
        str(CHECKPOINT_DIR / "bc_sac" / "best" / "best_model.zip"),
        str(CHECKPOINT_DIR / "bc_sac" / "final.zip"),
    ]
    model = None
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading {path}...")
            model = SAC.load(path, device=device)
            break
    if model is None:
        print("No BC+SAC checkpoint found")
        return

    from rl_train import evaluate_model, print_comparison
    result = evaluate_model(model, "bc_sac", pc, cc, n_episodes=n_episodes)
    print_comparison([result])
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC → RL parking")
    parser.add_argument("--n-demos", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--parking-type", default="perpendicular",
                        choices=["perpendicular", "parallel"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bc-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = _default_device()

    pc = ParkingConfig(parking_type=args.parking_type)
    cc = CarConfig()

    if args.eval_only:
        evaluate_bc_sac(pc, cc, device=args.device)
    elif args.bc_only:
        transitions = collect_dense_demos(pc, cc, n_demos=args.n_demos)
        if transitions:
            obs_dim = transitions[0]["obs"].shape[0]
            act_dim = transitions[0]["action"].shape[0]
            bc_policy = train_bc(
                transitions, obs_dim, act_dim,
                epochs=args.bc_epochs, device=args.device,
            )
            _evaluate_bc(bc_policy, pc, cc, args.device, n_episodes=100)
    else:
        model = train_bc_then_rl(
            pc, cc,
            n_demos=args.n_demos,
            bc_epochs=args.bc_epochs,
            rl_timesteps=args.timesteps,
            seed=args.seed,
            device=args.device,
        )

        print("\n" + "=" * 60)
        print("  Final Evaluation")
        print("=" * 60)
        evaluate_bc_sac(pc, cc, n_episodes=100, device=args.device)
