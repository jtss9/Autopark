"""
Residual RL for parking: Hybrid A* provides the base action, SAC learns
a small correction on top.

The agent observes (env_obs, base_action) and outputs a residual δa ∈ [-α, α].
The executed action is clip(a_base + δa, -1, 1). This guarantees a reasonable
baseline policy from step 0 and lets RL focus on fine corrections.

Usage:
  python src/rl_residual.py --timesteps 1000000 --device cuda
  python src/rl_residual.py --eval-only
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as e:
    raise ImportError("pip install -r requirements-rl.txt") from e

from config import CarConfig, ParkingConfig
from controller import CarDynamics
from geom import angle_diff, wrap_pi
from hybrid_astar import OccupancyGrid, perpendicular_goal_pose, parallel_goal_pose
from parking_lot import ParkingLot
from rl_env import ParkingEnv
from trajectory import plan_trajectory, Waypoint

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LOG_DIR = Path(__file__).parent.parent / "logs"


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _interpolate_waypoints(waypoints: List[Waypoint], dt: float, max_speed: float) -> List[Waypoint]:
    """Densify A* waypoints into per-timestep targets by linear interpolation."""
    dense = [waypoints[0]]
    for i in range(1, len(waypoints)):
        prev, curr = waypoints[i - 1], waypoints[i]
        dx, dy = curr.x - prev.x, curr.y - prev.y
        dist = math.hypot(dx, dy)
        fwd = dx * math.cos(prev.theta) + dy * math.sin(prev.theta)
        speed = max_speed * 0.15 if fwd >= 0 else -max_speed * 0.15
        n_steps = max(1, int(dist / (abs(speed) * dt)))
        for s in range(1, n_steps + 1):
            t = s / n_steps
            x = prev.x + t * dx
            y = prev.y + t * dy
            theta = prev.theta + t * angle_diff(curr.theta, prev.theta)
            dense.append(Waypoint(x, y, wrap_pi(theta)))
    return dense


class ResidualParkingEnv(gym.Env):
    """Wraps ParkingEnv with an A*-based base controller.

    Observation: [env_obs (23), base_steer (1), base_speed (1), progress (1)] = 26-dim
    Action: residual δa ∈ [-residual_scale, residual_scale]² (mapped to [-1,1] for SAC)
    Executed: clip(a_base + residual_scale * δa, -1, 1)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        parking_config: Optional[ParkingConfig] = None,
        car_config: Optional[CarConfig] = None,
        max_episode_steps: int = 600,
        residual_scale: float = 0.3,
        dt: float = 0.1,
        max_speed: float = 2.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.pc = parking_config or ParkingConfig()
        self.cc = car_config or CarConfig()
        self.residual_scale = residual_scale
        self.max_episode_steps = max_episode_steps

        self._inner = ParkingEnv(
            parking_config=self.pc,
            car_config=self.cc,
            max_episode_steps=max_episode_steps,
            dt=dt,
            max_speed=max_speed,
            seed=seed,
        )

        self._plan_base_trajectory()

        # Action: residual correction in [-1, 1] (scaled by residual_scale)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation: inner_obs (23) + base_steer (1) + base_speed (1) + progress (1) = 26
        inner_dim = self._inner.observation_space.shape[0]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (inner_dim + 3,), np.float32
        )

        self._step_count = 0
        self._wp_idx = 0

    def _plan_base_trajectory(self):
        """Pre-compute the A* reference trajectory and densify it."""
        plan_pc = ParkingConfig(
            parking_type=self.pc.parking_type,
            planner="hybrid_astar",
            lane_width=self.pc.lane_width,
            spot_length=self.pc.spot_length,
            spot_width=self.pc.spot_width,
        )
        result = plan_trajectory(plan_pc, self.cc)
        if result.feasible:
            self._ref_wps = _interpolate_waypoints(
                result.waypoints, self._inner.dt, self._inner.max_speed
            )
        else:
            self._ref_wps = [Waypoint(*self._inner.lot.car_start_pose)]

        # Precompute forward/reverse per segment
        self._seg_reverse = []
        for i in range(1, len(self._ref_wps)):
            dx = self._ref_wps[i].x - self._ref_wps[i - 1].x
            dy = self._ref_wps[i].y - self._ref_wps[i - 1].y
            fwd = (dx * math.cos(self._ref_wps[i - 1].theta)
                   + dy * math.sin(self._ref_wps[i - 1].theta))
            self._seg_reverse.append(fwd < -0.001)

    def _base_action(self) -> Tuple[float, float]:
        """Compute the A* base controller action for the current state."""
        if self._wp_idx >= len(self._ref_wps):
            return 0.0, 0.0

        cx, cy, ct = self._inner._car.x, self._inner._car.y, self._inner._car.theta

        # Advance waypoint index past reached points
        while self._wp_idx < len(self._ref_wps) - 1:
            wp = self._ref_wps[self._wp_idx]
            if math.hypot(wp.x - cx, wp.y - cy) > 0.3:
                break
            self._wp_idx += 1

        # Lookahead target
        lookahead_idx = min(self._wp_idx + 3, len(self._ref_wps) - 1)
        target = self._ref_wps[lookahead_idx]

        dx = target.x - cx
        dy = target.y - cy
        dist = math.hypot(dx, dy)

        is_rev = (self._wp_idx < len(self._seg_reverse)
                  and self._seg_reverse[self._wp_idx])

        # Transform to local frame
        local_x = dx * math.cos(ct) + dy * math.sin(ct)
        local_y = -dx * math.sin(ct) + dy * math.cos(ct)

        if is_rev:
            local_x = -local_x
            local_y = -local_y

        # Pure pursuit steering
        if abs(local_x) > 0.01:
            curvature = 2.0 * local_y / (local_x ** 2 + local_y ** 2)
            steer_angle = math.atan(curvature * self.cc.wheelbase)
        else:
            steer_angle = math.copysign(self.cc.max_steer * 0.3, local_y)

        steer = float(np.clip(steer_angle / self.cc.max_steer, -1.0, 1.0))

        # Speed: slow, proportional to distance, with approach slowdown
        remaining = len(self._ref_wps) - 1 - self._wp_idx
        approach = min(1.0, remaining / 10.0)
        steer_slow = 1.0 - 0.5 * abs(steer)
        speed_mag = float(np.clip(dist * 0.5 * steer_slow * approach, 0.05, 0.25))
        speed = -speed_mag if is_rev else speed_mag

        return steer, speed

    def _make_obs(self, inner_obs: np.ndarray, base_steer: float, base_speed: float) -> np.ndarray:
        progress = self._wp_idx / max(1, len(self._ref_wps) - 1)
        return np.concatenate([
            inner_obs,
            np.array([base_steer, base_speed, progress], dtype=np.float32),
        ])

    def reset(self, *, seed=None, options=None):
        if options is None:
            options = {"start_pose": self._inner.lot.car_start_pose}
        inner_obs, info = self._inner.reset(seed=seed, options=options)
        self._step_count = 0
        self._wp_idx = 0
        base_steer, base_speed = self._base_action()
        obs = self._make_obs(inner_obs, base_steer, base_speed)
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Compute base action from A* tracker
        base_steer, base_speed = self._base_action()

        # Apply residual: action ∈ [-1,1] scaled by residual_scale
        residual_steer = float(action[0]) * self.residual_scale
        residual_speed = float(action[1]) * self.residual_scale

        final_steer = float(np.clip(base_steer + residual_steer, -1.0, 1.0))
        final_speed = float(np.clip(base_speed + residual_speed, -1.0, 1.0))

        # Step inner env with the combined action
        final_action = np.array([final_steer, final_speed], dtype=np.float32)
        inner_obs, reward, terminated, truncated, info = self._inner.step(final_action)

        # Add bonus for following the reference closely
        if self._wp_idx < len(self._ref_wps):
            ref = self._ref_wps[min(self._wp_idx, len(self._ref_wps) - 1)]
            track_err = math.hypot(
                self._inner._car.x - ref.x,
                self._inner._car.y - ref.y,
            )
            reward += max(0, 0.5 - track_err) * 0.2

        # Small penalty for large residuals (encourage staying close to base)
        reward -= 0.1 * (abs(residual_steer) + abs(residual_speed))

        info["base_steer"] = base_steer
        info["base_speed"] = base_speed
        info["residual_steer"] = residual_steer
        info["residual_speed"] = residual_speed
        info["wp_progress"] = self._wp_idx / max(1, len(self._ref_wps) - 1)

        # Recompute base action for next obs
        next_base_steer, next_base_speed = self._base_action()
        obs = self._make_obs(inner_obs, next_base_steer, next_base_speed)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self._inner.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_residual(
    pc: ParkingConfig, cc: CarConfig,
    total_timesteps: int = 1_000_000,
    residual_scale: float = 0.3,
    seed: int = 0, verbose: int = 1, device: str = "auto",
) -> SAC:
    if device == "auto":
        device = _default_device()

    env = ResidualParkingEnv(
        parking_config=pc, car_config=cc,
        residual_scale=residual_scale, seed=seed,
    )
    env = Monitor(env)

    eval_env = ResidualParkingEnv(
        parking_config=pc, car_config=cc,
        residual_scale=residual_scale, seed=seed + 1000,
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(LOG_DIR / "residual" / "tensorboard"),
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    save_dir = str(CHECKPOINT_DIR / "residual" / "best")
    os.makedirs(save_dir, exist_ok=True)

    eval_cb = EvalCallback(
        Monitor(eval_env),
        best_model_save_path=save_dir,
        log_path=str(LOG_DIR / "residual" / "eval"),
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    save_path = str(CHECKPOINT_DIR / "residual" / "final")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[Residual SAC] Training for {total_timesteps} steps on {device}...")
    print(f"  residual_scale={residual_scale}")
    model.learn(total_timesteps=total_timesteps, callback=[eval_cb])
    model.save(save_path)
    print(f"[Residual SAC] Saved to {save_path}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_residual(
    pc: ParkingConfig, cc: CarConfig,
    n_episodes: int = 100, device: str = "auto",
):
    if device == "auto":
        device = _default_device()

    candidates = [
        str(CHECKPOINT_DIR / "residual" / "best" / "best_model.zip"),
        str(CHECKPOINT_DIR / "residual" / "final.zip"),
    ]
    model = None
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading {path}...")
            model = SAC.load(path, device=device)
            break
    if model is None:
        print("No residual checkpoint found")
        return None

    env = ResidualParkingEnv(parking_config=pc, car_config=cc)

    results = {
        "successes": 0, "collisions": 0, "timeouts": 0,
        "total_reward": 0.0, "total_steps": 0,
        "fixed_success": 0, "fixed_total": 0,
        "random_success": 0, "random_total": 0,
        "avg_residual_mag": 0.0,
    }
    total_res_mag = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        use_fixed = (ep % 2 == 0)
        if use_fixed:
            obs, _ = env.reset(options={"start_pose": env._inner.lot.car_start_pose})
            results["fixed_total"] += 1
        else:
            obs, _ = env.reset(options=None)
            results["random_total"] += 1

        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            total_res_mag += abs(info.get("residual_steer", 0)) + abs(info.get("residual_speed", 0))
            total_steps += 1
            done = terminated or truncated

        results["total_reward"] += ep_reward
        results["total_steps"] += total_steps

        if info.get("is_success", False):
            results["successes"] += 1
            if use_fixed:
                results["fixed_success"] += 1
            else:
                results["random_success"] += 1
        elif info.get("collision", False):
            results["collisions"] += 1
        else:
            results["timeouts"] += 1

    n = n_episodes
    results["success_rate"] = results["successes"] / n
    results["collision_rate"] = results["collisions"] / n
    results["avg_reward"] = results["total_reward"] / n
    results["avg_residual_mag"] = total_res_mag / max(total_steps, 1)

    fs = results["fixed_success"] / max(results["fixed_total"], 1)
    rs = results["random_success"] / max(results["random_total"], 1)

    print(f"\n{'='*60}")
    print(f"  Residual RL Evaluation ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Success:   {results['success_rate']:.1%} "
          f"(fixed={fs:.1%}, random={rs:.1%})")
    print(f"  Collision: {results['collision_rate']:.1%}")
    print(f"  Avg reward: {results['avg_reward']:.1f}")
    print(f"  Avg |residual|: {results['avg_residual_mag']:.4f}")
    print(f"{'='*60}")
    return results


# ---------------------------------------------------------------------------
# Also evaluate the base controller alone (residual = 0)
# ---------------------------------------------------------------------------

def evaluate_base_only(
    pc: ParkingConfig, cc: CarConfig,
    n_episodes: int = 100,
):
    """Evaluate the A* base controller without any RL correction."""
    env = ResidualParkingEnv(parking_config=pc, car_config=cc)

    successes = 0
    collisions = 0
    fixed_suc = 0
    fixed_tot = 0

    for ep in range(n_episodes):
        use_fixed = (ep % 2 == 0)
        if use_fixed:
            obs, _ = env.reset(options={"start_pose": env._inner.lot.car_start_pose})
            fixed_tot += 1
        else:
            obs, _ = env.reset(options=None)

        done = False
        while not done:
            # Zero residual = pure base controller
            obs, _, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            done = terminated or truncated

        if info.get("is_success", False):
            successes += 1
            if use_fixed:
                fixed_suc += 1
        elif info.get("collision", False):
            collisions += 1

    print(f"\n  Base controller (no RL): {successes}/{n_episodes} success "
          f"({successes/n_episodes:.1%}), "
          f"fixed={fixed_suc}/{fixed_tot}, "
          f"collision={collisions}/{n_episodes}")
    return successes / n_episodes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual RL parking")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--parking-type", default="perpendicular",
                        choices=["perpendicular", "parallel"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = _default_device()

    pc = ParkingConfig(parking_type=args.parking_type)
    cc = CarConfig()

    if args.base_only:
        evaluate_base_only(pc, cc, n_episodes=100)
    elif args.eval_only:
        evaluate_residual(pc, cc, device=args.device)
    else:
        # First show base controller performance
        print("=" * 60)
        print("  Evaluating base A* controller (no RL)")
        print("=" * 60)
        evaluate_base_only(pc, cc, n_episodes=50)

        # Then train residual
        print()
        model = train_residual(
            pc, cc,
            total_timesteps=args.timesteps,
            residual_scale=args.residual_scale,
            seed=args.seed,
            device=args.device,
        )

        # Final eval
        print()
        evaluate_residual(pc, cc, n_episodes=100, device=args.device)
