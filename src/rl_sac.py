"""
SAC-based parking planner: training, evaluation, and trajectory generation.

Wraps stable-baselines3 SAC with curriculum learning. Produces TrajectoryResult
objects compatible with the existing evaluator and simulator.

On Apple Silicon Macs, training uses MPS (Metal Performance Shaders) by default
for GPU acceleration. Pass device="cpu" to force CPU if needed.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    raise ImportError(
        "RL training requires stable-baselines3 and torch. "
        "Install with: pip install -r requirements-rl.txt"
    ) from e

import torch

from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff, path_length as _path_length
from hybrid_astar import OccupancyGrid, perpendicular_goal_pose, parallel_goal_pose
from parking_lot import ParkingLot
from rl_env import ParkingEnv, CURRICULUM_STAGES
from trajectory import TrajectoryResult, Waypoint


def _default_device() -> str:
    """Pick the best available device: MPS on Apple Silicon, CUDA if available, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LOG_DIR = Path(__file__).parent.parent / "logs"


class CurriculumCallback(BaseCallback):
    """Promotes the training environment to the next curriculum stage
    when the evaluation success rate exceeds the threshold."""

    def __init__(self, eval_env: ParkingEnv, check_freq: int = 10000, verbose: int = 0):
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


def train_sac(
    pc: Optional[ParkingConfig] = None,
    cc: Optional[CarConfig] = None,
    total_timesteps: int = 500_000,
    curriculum: bool = True,
    start_stage: int = 0,
    save_path: Optional[str] = None,
    seed: int = 0,
    verbose: int = 1,
    device: str = "auto",
) -> SAC:
    """Train a SAC policy for parking.

    Args:
        device: "auto" picks MPS on Apple Silicon, CUDA if available, else CPU.
                Pass "cpu" or "mps" or "cuda" to override.
    """
    pc = pc or ParkingConfig()
    cc = cc or CarConfig()
    if device == "auto":
        device = _default_device()

    env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=start_stage, seed=seed,
    )
    env = Monitor(env)

    eval_env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=start_stage, seed=seed + 1000,
    )

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
        tensorboard_log=str(LOG_DIR / "sac_tensorboard"),
    )

    callbacks = []
    if curriculum:
        callbacks.append(CurriculumCallback(eval_env, check_freq=10000, verbose=verbose))

    save_dir = save_path or str(CHECKPOINT_DIR / "sac_parking")
    os.makedirs(os.path.dirname(save_dir) if "/" in save_dir else save_dir, exist_ok=True)

    eval_cb = EvalCallback(
        Monitor(eval_env),
        best_model_save_path=str(CHECKPOINT_DIR / "best"),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
    )
    callbacks.append(eval_cb)

    print(f"Training SAC ({device}) for {total_timesteps} timesteps...")
    print(f"  Curriculum: {curriculum}, start_stage: {start_stage}")
    print(f"  Parking type: {pc.parking_type}")

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(save_dir)
    print(f"Model saved to {save_dir}")
    return model


def load_sac(path: str, device: str = "auto") -> SAC:
    """Load a trained SAC model."""
    if device == "auto":
        device = _default_device()
    return SAC.load(path, device=device)


def rollout_sac(
    model: SAC,
    pc: Optional[ParkingConfig] = None,
    cc: Optional[CarConfig] = None,
    max_steps: int = 300,
    deterministic: bool = True,
    curriculum_stage: int = 0,
    use_lot_start: bool = False,
) -> TrajectoryResult:
    """Generate a parking trajectory using a trained SAC policy.

    When use_lot_start=True, the car starts from ParkingLot.car_start_pose
    (the fixed position shown in the UI) instead of a random curriculum sample.
    """
    pc = pc or ParkingConfig()
    cc = cc or CarConfig()

    env = ParkingEnv(
        parking_config=pc, car_config=cc,
        curriculum_stage=curriculum_stage,
        max_episode_steps=max_steps,
    )

    reset_options = None
    if use_lot_start:
        reset_options = {"start_pose": env.lot.car_start_pose}
    obs, info = env.reset(options=reset_options)
    waypoints = [Waypoint(env._car.x, env._car.y, env._car.theta)]

    total_steering_change = 0.0
    gear_switches = 0
    collision = False
    success = False

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        waypoints.append(Waypoint(env._car.x, env._car.y, env._car.theta))

        if terminated or truncated:
            success = info.get("is_success", False)
            collision = info.get("collision", False)
            gear_switches = info.get("gear_switches", 0)
            break

    goal = env._goal_pose
    final = waypoints[-1]
    final_pos_error = math.hypot(final.x - goal[0], final.y - goal[1])
    final_heading_error = abs(_angle_diff(final.theta, goal[2]))
    total_path_length = _path_length(waypoints)

    # Steering smoothness
    for i in range(2, len(waypoints)):
        a, b, c = waypoints[i - 2], waypoints[i - 1], waypoints[i]
        da = _angle_diff(b.theta, a.theta)
        db = _angle_diff(c.theta, b.theta)
        total_steering_change += abs(db - da)

    if success:
        message = "SAC: successfully parked."
    elif collision:
        message = "SAC: collision during rollout."
    else:
        message = "SAC: exceeded step limit without parking."

    metrics = {
        "planner_kind": "sac",
        "path_length_m": total_path_length,
        "waypoints": len(waypoints),
        "final_pos_error_m": final_pos_error,
        "final_heading_error_deg": math.degrees(final_heading_error),
        "fully_in_spot": success,
        "episode_steps": len(waypoints) - 1,
        "collision": collision,
        "gear_switches": gear_switches,
        "steering_change_cost": total_steering_change,
        "obstacles": len(env._obstacles),
    }

    return TrajectoryResult(
        waypoints=waypoints,
        feasible=success,
        message=message,
        phase_starts=[0],
        phase_names=["SAC rollout"],
        metrics=metrics,
    )


def plan_sac(
    pc: ParkingConfig,
    cc: CarConfig,
    model_path: Optional[str] = None,
    device: str = "auto",
    attempts: int = 5,
) -> TrajectoryResult:
    """Plan a trajectory using a pre-trained SAC model.

    If no model_path is given, looks for the best checkpoint first, then the
    final checkpoint.  If no checkpoint exists, trains a new model.
    Tries multiple rollouts (with slight stochasticity) and returns the best.
    """
    if device == "auto":
        device = _default_device()

    # Prefer the best eval checkpoint over the final one
    candidates = []
    if model_path:
        candidates.append(model_path)
    candidates.extend([
        str(CHECKPOINT_DIR / "sac" / "best" / "best_model.zip"),
        str(CHECKPOINT_DIR / "best" / "best_model.zip"),
        str(CHECKPOINT_DIR / "sac" / "final.zip"),
        str(CHECKPOINT_DIR / "sac_parking.zip"),
    ])

    start_t = time.perf_counter()
    model = None
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading SAC model from {path}...", end="", flush=True)
            model = load_sac(path, device=device)
            break

    if model is None:
        print("No SAC checkpoint found. Training new model...")
        model = train_sac(pc=pc, cc=cc, device=device)

    best_result = None
    for i in range(attempts):
        deterministic = (i == 0)
        result = rollout_sac(
            model, pc=pc, cc=cc, use_lot_start=True,
            deterministic=deterministic,
        )
        if best_result is None or _result_score(result) > _result_score(best_result):
            best_result = result
        if result.feasible:
            break

    elapsed = time.perf_counter() - start_t
    best_result.metrics["planning_time_s"] = elapsed

    if best_result.feasible:
        print(f" done in {elapsed:.1f}s ({len(best_result.waypoints)} waypoints)")
    else:
        print(f" failed ({elapsed:.1f}s): {best_result.message}")

    return best_result


def _result_score(result: TrajectoryResult) -> float:
    """Higher is better.  Feasible results always beat infeasible ones."""
    if result.feasible:
        return 1000.0 - result.metrics.get("final_pos_error_m", 0)
    if result.metrics.get("collision"):
        return -100.0 + result.metrics.get("episode_steps", 0)
    return -result.metrics.get("final_pos_error_m", 999)


# ---------------------------------------------------------------------------
# CLI entrypoint for standalone training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SAC parking agent")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--parking-type", default="perpendicular",
                        choices=["perpendicular", "parallel"])
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', or 'cuda'")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    pc = ParkingConfig(parking_type=args.parking_type)
    cc = CarConfig()

    model = train_sac(
        pc=pc,
        cc=cc,
        total_timesteps=args.timesteps,
        curriculum=not args.no_curriculum,
        start_stage=args.stage,
        save_path=args.output,
        seed=args.seed,
        device=args.device,
    )

    print("\nEvaluating trained model...")
    result = rollout_sac(model, pc=pc, cc=cc)
    print(f"  Success: {result.feasible}")
    print(f"  Path length: {result.metrics['path_length_m']:.2f} m")
    print(f"  Final pos error: {result.metrics['final_pos_error_m']:.3f} m")
    print(f"  Final heading error: {result.metrics['final_heading_error_deg']:.1f} deg")
    print(f"  Steps: {result.metrics['episode_steps']}")
