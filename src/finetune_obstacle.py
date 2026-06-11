"""Obstacle-focused fine-tune for the perpendicular SAC policy.

The expanded model parks the base scene 100% from the fixed start but
fails it whenever an obstacle is present (the obstacle competence it has
lives on randomized starts). This script warm-starts the deployed
checkpoint and trains specifically on base-scene + single-obstacle
worlds, with a reverse curriculum along each obstacle world's reference
path, demo seeding, base-experience prefill, and checkpoint gating on a
mixed base/obstacle fixed-start metric (base double-weighted).

Usage:  python src/finetune_obstacle.py --timesteps 250000
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

import rl_env
from config import CarConfig, ParkingConfig
from rl_env import CurriculumStage, ParkingEnv, obstacle_candidates
from rl_train import (
    CurriculumCallback, FixedStartEvalCallback, _prefill_base_experience,
    collect_demonstrations, seed_replay_buffer,
)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LOG_DIR = Path(__file__).parent.parent / "logs"

BASE_SCENE = (6.0, 6.0, 2.5, 4.5, 1.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument(
        "--warm-start",
        default=str(CHECKPOINT_DIR / "sac" / "best_fixed" / "model.zip"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pc, cc = ParkingConfig(), CarConfig()

    # Focused curriculum: base scene only, one random feasible obstacle,
    # reverse curriculum along each obstacle world's reference path.
    rl_env.CURRICULUM_STAGES[:] = [
        CurriculumStage("obs_near", path_t_lo=0.5, fixed_start_ratio=0.2,
                        success_rate_threshold=0.5, random_obstacle=True),
        CurriculumStage("obs_full", path_t_lo=0.0, fixed_start_ratio=0.5,
                        success_rate_threshold=0.5, random_obstacle=True),
    ]

    feasible_obs = []
    probe = ParkingEnv(parking_config=pc, car_config=cc)
    for ob in obstacle_candidates("perpendicular", BASE_SCENE):
        if probe.set_world(BASE_SCENE, ob):
            feasible_obs.append(ob)
    print(f"[Finetune] {len(feasible_obs)} feasible base-scene obstacle worlds")

    env = Monitor(ParkingEnv(parking_config=pc, car_config=cc,
                             max_episode_steps=350, seed=args.seed))
    eval_env = ParkingEnv(parking_config=pc, car_config=cc,
                          max_episode_steps=450, seed=args.seed + 1000)

    print(f"[Finetune] Warm-starting from {args.warm_start}")
    model = SAC.load(args.warm_start, env=env, device="cpu",
                     tensorboard_log=str(LOG_DIR / "sac" / "tensorboard"))
    model.learning_starts = 0

    demo_worlds = [(None, None)] + [(BASE_SCENE, ob) for ob in feasible_obs]
    demos = collect_demonstrations(pc, cc, n_demos=8 * len(demo_worlds),
                                   worlds=demo_worlds)
    if demos:
        seed_replay_buffer(model, demos)
        print(f"[Finetune] Seeded {len(demos)} demo transitions")

    _prefill_base_experience(model, pc, cc, n_steps=20_000)

    # Checkpoint gating on mixed fixed-start metric, base double-weighted so
    # obstacle gains cannot silently trade away the deployment scenario.
    eval_worlds = [(None, None), (None, None)] + \
        [(BASE_SCENE, ob) for ob in feasible_obs[:3]]
    fixed_cb = FixedStartEvalCallback(
        pc, cc,
        save_path=str(CHECKPOINT_DIR / "sac" / "best_fixed" / "model"),
        eval_freq=15_000, n_episodes=4, verbose=1, eval_worlds=eval_worlds,
    )
    cur_cb = CurriculumCallback(eval_env, check_freq=15_000, verbose=1)

    print(f"[Finetune] Training {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps, callback=[cur_cb, fixed_cb],
                reset_num_timesteps=False)
    model.save(str(CHECKPOINT_DIR / "sac" / "final_obstacle"))
    print("[Finetune] done")


if __name__ == "__main__":
    main()
