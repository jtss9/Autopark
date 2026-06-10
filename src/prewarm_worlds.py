"""Pre-compute and persist hybrid A* reference paths for all randomized
training worlds (scene pool + single-obstacle variants).

Feasibility checks for randomized worlds are the slow part of training
startup — a *failed* search (infeasible world) can take up to a minute.
Running this once populates checkpoints/refpath_cache.json so training and
evaluation never stall on first-time planning.

Usage:  python src/prewarm_worlds.py
"""
import time

from config import CarConfig, ParkingConfig
from rl_env import ParkingEnv, scene_pool, obstacle_candidates


def main() -> None:
    for ptype in ("perpendicular", "parallel"):
        pc = ParkingConfig(parking_type=ptype)
        env = ParkingEnv(parking_config=pc, car_config=CarConfig())

        pool = scene_pool(ptype)
        worlds = [(scene, None) for scene in pool]
        # Obstacle worlds draw from the same scene subset the curriculum uses.
        for scene in pool[:8]:
            for obstacle in obstacle_candidates(ptype, scene):
                worlds.append((scene, obstacle))

        t0 = time.perf_counter()
        feasible = 0
        for i, (scene, obstacle) in enumerate(worlds):
            ok = env.set_world(scene, obstacle)
            feasible += ok
            print(f"[{ptype}] {i + 1}/{len(worlds)} "
                  f"{'ok' if ok else 'INFEASIBLE'} scene={scene} "
                  f"obstacle={obstacle}", flush=True)
        print(f"[{ptype}] done: {feasible}/{len(worlds)} feasible worlds, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
