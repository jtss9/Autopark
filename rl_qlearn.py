"""
Tabular Q-learning parking planner.

This module trains a small tabular Q-learning agent on the parking scene and
returns the resulting greedy rollout as a `TrajectoryResult`, so it slots into
the same evaluator and simulator as the geometric/MPC baseline and Hybrid A*.

The point is NOT to beat Hybrid A* — tabular RL famously struggles with
continuous SE(2) navigation without function approximation. It is a *learned*
alternative used as a comparison baseline in the report, supporting the
narrative that search and sampling-based planners outperform pure learning on
this class of small, geometric, sparse-reward problems.

State    : (ix, iy, ith) integer grid bucket around the rear-axle pose.
Action   : (gear, steer_index) where gear ∈ {+1, -1} and steer comes from a
           small set, so the discrete action set has 10 entries.
Reward   : potential-based shaping on distance + heading toward the spot,
           with a large positive bonus on parking success and a large
           negative penalty on collision.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff
from hybrid_astar import (
    OccupancyGrid,
    parallel_goal_pose,
    perpendicular_goal_pose,
)
from parking_lot import ParkingLot
from scenarios import obstacles_for
from trajectory import TrajectoryResult, Waypoint


Pose = Tuple[float, float, float]
StateKey = Tuple[int, int, int]


@dataclass
class QLearnConfig:
    # Discretisation
    xy_resolution: float = 0.45
    theta_bins: int = 12

    # Dynamics
    action_step: float = 0.7
    action_repeat: int = 2

    # Training schedule
    episodes: int = 40000
    max_steps_per_episode: int = 50
    train_time_budget_s: float = 30.0
    seed: int = 0

    # Learning hyperparameters
    alpha: float = 0.4
    gamma: float = 0.96
    epsilon_start: float = 0.95
    epsilon_end: float = 0.05
    optimistic_init: float = 1.0

    # Reward shaping
    step_penalty: float = -0.05
    dist_reward_gain: float = 1.5
    heading_reward_gain: float = 0.4
    collision_penalty: float = -10.0
    goal_bonus: float = 25.0

    # Acceptance tolerance for goal
    goal_xy_tol: float = 0.45
    goal_theta_tol_deg: float = 15.0

    # Curriculum: probability of starting an episode near the goal so the
    # agent learns the terminal phase before the full approach.
    curriculum_start_prob: float = 0.35
    curriculum_radius: float = 2.5


class QLearningPlanner:
    def __init__(
        self,
        lot: ParkingLot,
        cc: CarConfig,
        grid: OccupancyGrid,
        cfg: Optional[QLearnConfig] = None,
    ):
        self.lot = lot
        self.cc = cc
        self.grid = grid
        self.cfg = cfg or QLearnConfig()

        steer_max = cc.max_steer
        self.actions: List[Tuple[int, float]] = [
            (gear, steer)
            for gear in (1, -1)
            for steer in (-steer_max, -0.5 * steer_max, 0.0, 0.5 * steer_max, steer_max)
        ]
        self.Q: Dict[StateKey, List[float]] = {}
        self.rng = random.Random(self.cfg.seed)

        # Diagnostic state populated by train()
        self.episodes_run = 0
        self.successful_episodes = 0
        self.training_time_s = 0.0

    # ------------------------------------------------------------------
    # Discretisation helpers
    # ------------------------------------------------------------------
    def _state_key(self, pose: Pose) -> StateKey:
        ix = int(round(pose[0] / self.cfg.xy_resolution))
        iy = int(round(pose[1] / self.cfg.xy_resolution))
        ith = int(
            round(((pose[2] + math.pi) % (2 * math.pi)) / (2 * math.pi) * self.cfg.theta_bins)
        ) % self.cfg.theta_bins
        return (ix, iy, ith)

    def _q_row(self, key: StateKey) -> List[float]:
        row = self.Q.get(key)
        if row is None:
            row = [self.cfg.optimistic_init] * len(self.actions)
            self.Q[key] = row
        return row

    # ------------------------------------------------------------------
    # Dynamics + reward
    # ------------------------------------------------------------------
    def _step(self, pose: Pose, gear: int, steer: float) -> Pose:
        x, y, theta = pose
        ds = self.cfg.action_step
        # Mid-point integration for accuracy without too many sub-steps.
        for _ in range(self.cfg.action_repeat):
            half = ds / self.cfg.action_repeat
            dtheta = gear * half * math.tan(steer) / self.cc.wheelbase
            x += gear * half * math.cos(theta + 0.5 * dtheta)
            y += gear * half * math.sin(theta + 0.5 * dtheta)
            theta = _angle_diff(theta + dtheta, 0.0)
        return (x, y, theta)

    def _is_goal(self, pose: Pose, goal: Pose) -> bool:
        dx = pose[0] - goal[0]
        dy = pose[1] - goal[1]
        if math.hypot(dx, dy) > self.cfg.goal_xy_tol:
            return False
        if abs(_angle_diff(pose[2], goal[2])) > math.radians(self.cfg.goal_theta_tol_deg):
            return False
        return self.grid.pose_is_fully_in_spot(pose, margin=0.02)

    def _shaping(self, pose: Pose, next_pose: Pose, goal: Pose) -> float:
        old_d = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        new_d = math.hypot(next_pose[0] - goal[0], next_pose[1] - goal[1])
        old_h = abs(_angle_diff(pose[2], goal[2]))
        new_h = abs(_angle_diff(next_pose[2], goal[2]))
        return (
            self.cfg.dist_reward_gain * (old_d - new_d)
            + self.cfg.heading_reward_gain * (old_h - new_h)
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _sample_episode_start(
        self,
        start: Pose,
        goal: Pose,
        progress: float,
    ) -> Pose:
        """Reverse-curriculum sampling.

        Progress goes 0 → 1 across training. Early on, sample episode starts
        near the goal so the agent learns terminal-phase Q values. Later,
        expand the radius and increasingly use the true start so the agent
        learns the full approach. This is a standard trick for sparse-reward
        navigation problems.
        """
        cfg = self.cfg
        if self.rng.random() < (1.0 - cfg.curriculum_start_prob) * progress:
            return start
        max_radius = math.hypot(start[0] - goal[0], start[1] - goal[1]) + 1.0
        radius_cap = cfg.curriculum_radius + progress * (max_radius - cfg.curriculum_radius)
        for _ in range(25):
            radius = self.rng.uniform(0.3, max(0.4, radius_cap))
            angle = self.rng.uniform(-math.pi, math.pi)
            x = goal[0] + radius * math.cos(angle)
            y = goal[1] + radius * math.sin(angle)
            theta = goal[2] + self.rng.uniform(-math.pi / 3, math.pi / 3)
            theta = _angle_diff(theta, 0.0)
            if self.grid.pose_is_valid((x, y, theta)):
                return (x, y, theta)
        return start

    def train(self, start: Pose, goal: Pose) -> None:
        cfg = self.cfg
        epsilon = cfg.epsilon_start
        decay = (cfg.epsilon_end / cfg.epsilon_start) ** (1.0 / max(1, cfg.episodes))
        start_t = time.perf_counter()

        for ep in range(cfg.episodes):
            self.episodes_run = ep + 1
            if time.perf_counter() - start_t > cfg.train_time_budget_s:
                break

            progress = ep / max(1, cfg.episodes - 1)
            pose = self._sample_episode_start(start, goal, progress)
            for _ in range(cfg.max_steps_per_episode):
                key = self._state_key(pose)
                row = self._q_row(key)
                if self.rng.random() < epsilon:
                    a_idx = self.rng.randrange(len(self.actions))
                else:
                    a_idx = max(range(len(self.actions)), key=row.__getitem__)

                gear, steer = self.actions[a_idx]
                next_pose = self._step(pose, gear, steer)

                if not self.grid.pose_is_valid(next_pose):
                    reward = cfg.collision_penalty
                    row[a_idx] += cfg.alpha * (reward - row[a_idx])
                    break

                reward = cfg.step_penalty + self._shaping(pose, next_pose, goal)
                done = self._is_goal(next_pose, goal)
                if done:
                    reward += cfg.goal_bonus
                    self.successful_episodes += 1

                next_row = self._q_row(self._state_key(next_pose))
                target = reward + (0.0 if done else cfg.gamma * max(next_row))
                row[a_idx] += cfg.alpha * (target - row[a_idx])

                if done:
                    break
                pose = next_pose

            epsilon = max(cfg.epsilon_end, epsilon * decay)

        self.training_time_s = time.perf_counter() - start_t

    # ------------------------------------------------------------------
    # Greedy rollout (the "planned" path)
    # ------------------------------------------------------------------
    def rollout(
        self,
        start: Pose,
        goal: Pose,
        max_steps: int = 300,
        epsilon: float = 0.05,
    ) -> Tuple[List[Waypoint], bool, str]:
        """Roll out the learned policy.

        A small epsilon (default 5%) lets the rollout occasionally take a
        random action so it can escape local Q-plateaus that would otherwise
        trap a strictly greedy policy in the same cell pair. Loop detection
        still aborts on persistent oscillation.
        """
        path: List[Waypoint] = [Waypoint(*start)]
        pose = start
        visited: Dict[StateKey, int] = {}
        for step in range(max_steps):
            key = self._state_key(pose)
            visited[key] = visited.get(key, 0) + 1
            if visited[key] > 6:
                return path, False, "Q-learn: policy looping, no path."
            row = self.Q.get(key)
            if row is None:
                return path, False, "Q-learn: state unseen during training."
            if self.rng.random() < epsilon:
                a_idx = self.rng.randrange(len(self.actions))
            else:
                a_idx = max(range(len(self.actions)), key=row.__getitem__)
            gear, steer = self.actions[a_idx]
            next_pose = self._step(pose, gear, steer)
            if not self.grid.pose_is_valid(next_pose):
                # Random action hit a wall; try the greedy choice instead
                a_idx = max(range(len(self.actions)), key=row.__getitem__)
                gear, steer = self.actions[a_idx]
                next_pose = self._step(pose, gear, steer)
                if not self.grid.pose_is_valid(next_pose):
                    return path, False, "Q-learn: greedy rollout hit obstacle."
            path.append(Waypoint(*next_pose))
            pose = next_pose
            if self._is_goal(pose, goal):
                return path, True, "Q-learn: greedy rollout reached goal."
        return path, False, "Q-learn: greedy rollout exceeded step limit."


def plan_qlearn(
    pc: ParkingConfig,
    cc: CarConfig,
    cfg: Optional[QLearnConfig] = None,
) -> TrajectoryResult:
    lot = ParkingLot(pc, cc)
    obstacles = obstacles_for(lot)
    grid = OccupancyGrid(lot, obstacles=obstacles)

    if pc.parking_type == "perpendicular":
        goal = perpendicular_goal_pose(lot)
    elif pc.parking_type == "parallel":
        goal = parallel_goal_pose(lot)
    else:
        return TrajectoryResult(
            [], False, f"Q-learn: unknown parking type {pc.parking_type!r}.")

    planner = QLearningPlanner(lot, cc, grid, cfg)
    train_t = time.perf_counter()
    planner.train(lot.car_start_pose, goal)
    train_elapsed = time.perf_counter() - train_t

    rollout_t = time.perf_counter()
    path, ok, message = planner.rollout(lot.car_start_pose, goal)
    rollout_elapsed = time.perf_counter() - rollout_t

    final = path[-1] if path else Waypoint(*lot.car_start_pose)
    final_pos_error = math.hypot(final.x - goal[0], final.y - goal[1])
    final_heading_error = abs(_angle_diff(final.theta, goal[2]))
    path_length = sum(
        math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:])
    )
    fully_in_spot = grid.pose_is_fully_in_spot((final.x, final.y, final.theta), margin=0.0)
    success = ok and fully_in_spot
    final_message = message
    if ok and not fully_in_spot:
        final_message = "Q-learn: greedy rollout did not place car fully in spot."

    metrics = {
        "planning_time_s": train_elapsed + rollout_elapsed,
        "training_time_s": train_elapsed,
        "rollout_time_s": rollout_elapsed,
        "episodes": planner.episodes_run,
        "successful_episodes": planner.successful_episodes,
        "iterations": planner.episodes_run,
        "expanded_states": len(planner.Q),
        "path_length_m": path_length,
        "waypoints": len(path),
        "final_pos_error_m": final_pos_error,
        "final_heading_error_deg": math.degrees(final_heading_error),
        "fully_in_spot": fully_in_spot,
        "obstacles": len(obstacles),
        "planner_kind": "qlearn",
    }
    return TrajectoryResult(
        waypoints=path,
        feasible=success,
        message=final_message,
        phase_starts=[0],
        phase_names=["Q-learning rollout"],
        metrics=metrics,
    )
