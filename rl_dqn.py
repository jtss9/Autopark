"""
Deep Q-Network (DQN) parking planner.

Replaces the flat Q-table of rl_qlearn.py with a small neural network
that maps continuous, goal-relative state features to Q-values over the
same discrete action set. This removes the state-discretisation
bottleneck: the network generalises across nearby poses rather than
treating each grid cell as an independent entry.

Key DQN ingredients:
  - Experience-replay buffer (uniform sampling)
  - Target network updated every C steps
  - Epsilon-greedy exploration with decay
  - Goal-relative state features (7-dim): distance, heading error,
    longitudinal/lateral offset in the goal frame, absolute heading,
    x-fraction, y-fraction.

Requires PyTorch (CPU is sufficient). The evaluator skips this planner
gracefully if torch is not installed.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff, path_length as _path_length
from hybrid_astar import (
    OccupancyGrid,
    parallel_goal_pose,
    perpendicular_goal_pose,
)
from parking_lot import ParkingLot
from scenarios import obstacles_for
from trajectory import TrajectoryResult, Waypoint

Pose = Tuple[float, float, float]

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install with "
            "`pip install torch` (CPU-only is fine)."
        )


# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------
def _build_net(state_dim: int, n_actions: int):
    require_torch()
    return nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, n_actions),
    )


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass
class DQNConfig:
    action_step: float = 0.7
    action_repeat: int = 2

    episodes: int = 8000
    max_steps_per_episode: int = 50
    train_time_budget_s: float = 60.0
    seed: int = 0

    lr: float = 3e-4
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 5000
    batch_size: int = 32
    replay_capacity: int = 80_000
    target_update_freq: int = 300

    step_penalty: float = -0.05
    dist_reward_gain: float = 1.5
    heading_reward_gain: float = 0.4
    collision_penalty: float = -10.0
    goal_bonus: float = 25.0

    goal_xy_tol: float = 0.45
    goal_theta_tol_deg: float = 15.0

    curriculum_start_prob: float = 0.35
    curriculum_radius: float = 2.5


# ------------------------------------------------------------------
# Planner
# ------------------------------------------------------------------
class DQNPlanner:
    STATE_DIM = 7

    def __init__(
        self,
        lot: ParkingLot,
        cc: CarConfig,
        grid: OccupancyGrid,
        cfg: Optional[DQNConfig] = None,
    ):
        require_torch()
        self.lot = lot
        self.cc = cc
        self.grid = grid
        self.cfg = cfg or DQNConfig()
        self.goal: Pose = (0.0, 0.0, 0.0)

        steer_max = cc.max_steer
        self.actions: List[Tuple[int, float]] = [
            (gear, steer)
            for gear in (1, -1)
            for steer in (
                -steer_max, -0.5 * steer_max, 0.0,
                0.5 * steer_max, steer_max,
            )
        ]
        n_actions = len(self.actions)

        self.policy_net = _build_net(self.STATE_DIM, n_actions)
        self.target_net = _build_net(self.STATE_DIM, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=self.cfg.lr)
        self.replay = deque(maxlen=self.cfg.replay_capacity)
        self.rng = random.Random(self.cfg.seed)

        self.episodes_run = 0
        self.successful_episodes = 0
        self.training_time_s = 0.0
        self._update_count = 0

    # ------------------------------------------------------------------
    # State features
    # ------------------------------------------------------------------
    def _state_features(self, pose: Pose) -> "torch.Tensor":
        g = self.goal
        dx, dy = g[0] - pose[0], g[1] - pose[1]
        dist = math.hypot(dx, dy)
        herr = _angle_diff(pose[2], g[2])
        gcos, gsin = math.cos(g[2]), math.sin(g[2])
        longitudinal = dx * gcos + dy * gsin
        lateral = -dx * gsin + dy * gcos
        x_frac = pose[0] / max(1, self.lot.scene_w)
        y_frac = pose[1] / max(1, self.lot.scene_h)
        return torch.tensor(
            [dist, herr, longitudinal, lateral, pose[2], x_frac, y_frac],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _step(self, pose: Pose, gear: int, steer: float) -> Pose:
        x, y, theta = pose
        ds = self.cfg.action_step
        for _ in range(self.cfg.action_repeat):
            half = ds / self.cfg.action_repeat
            dtheta = gear * half * math.tan(steer) / self.cc.wheelbase
            x += gear * half * math.cos(theta + 0.5 * dtheta)
            y += gear * half * math.sin(theta + 0.5 * dtheta)
            theta = _angle_diff(theta + dtheta, 0.0)
        return (x, y, theta)

    def _is_goal(self, pose: Pose) -> bool:
        if math.hypot(pose[0] - self.goal[0],
                      pose[1] - self.goal[1]) > self.cfg.goal_xy_tol:
            return False
        if abs(_angle_diff(pose[2], self.goal[2])) > math.radians(
                self.cfg.goal_theta_tol_deg):
            return False
        return self.grid.pose_is_fully_in_spot(pose, margin=0.02)

    def _shaping(self, pose: Pose, nxt: Pose) -> float:
        g = self.goal
        cfg = self.cfg
        return (
            cfg.dist_reward_gain * (
                math.hypot(pose[0] - g[0], pose[1] - g[1])
                - math.hypot(nxt[0] - g[0], nxt[1] - g[1]))
            + cfg.heading_reward_gain * (
                abs(_angle_diff(pose[2], g[2]))
                - abs(_angle_diff(nxt[2], g[2])))
        )

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def _sample_start(self, start: Pose, progress: float) -> Pose:
        cfg = self.cfg
        if self.rng.random() < (1.0 - cfg.curriculum_start_prob) * progress:
            return start
        max_r = math.hypot(start[0] - self.goal[0],
                           start[1] - self.goal[1]) + 1.0
        cap = cfg.curriculum_radius + progress * (max_r - cfg.curriculum_radius)
        for _ in range(25):
            r = self.rng.uniform(0.3, max(0.4, cap))
            a = self.rng.uniform(-math.pi, math.pi)
            x = self.goal[0] + r * math.cos(a)
            y = self.goal[1] + r * math.sin(a)
            th = self.goal[2] + self.rng.uniform(-math.pi / 3, math.pi / 3)
            th = _angle_diff(th, 0.0)
            if self.grid.pose_is_valid((x, y, th)):
                return (x, y, th)
        return start

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _select_action(self, state: "torch.Tensor", epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(self.actions))
        with torch.no_grad():
            return int(self.policy_net(state.unsqueeze(0)).argmax(1).item())

    def _update(self) -> None:
        if len(self.replay) < self.cfg.batch_size:
            return
        batch = self.rng.sample(list(self.replay), self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.stack(states)
        a = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32)
        ns = torch.stack(next_states)
        d = torch.tensor(dones, dtype=torch.float32)

        q_vals = self.policy_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1).values
            target = r + self.cfg.gamma * next_q * (1 - d)

        loss = nn.functional.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, start: Pose, goal: Pose) -> None:
        self.goal = goal
        cfg = self.cfg
        t0 = time.perf_counter()
        eps = cfg.epsilon_start
        eps_step = ((cfg.epsilon_start - cfg.epsilon_end)
                    / max(1, cfg.epsilon_decay_episodes))
        total_steps = 0

        for ep in range(cfg.episodes):
            self.episodes_run = ep + 1
            if time.perf_counter() - t0 > cfg.train_time_budget_s:
                break

            progress = ep / max(1, cfg.episodes - 1)
            pose = self._sample_start(start, progress)
            state = self._state_features(pose)

            for _ in range(cfg.max_steps_per_episode):
                a_idx = self._select_action(state, eps)
                gear, steer = self.actions[a_idx]
                nxt = self._step(pose, gear, steer)
                total_steps += 1

                if not self.grid.pose_is_valid(nxt):
                    self.replay.append((
                        state, a_idx, cfg.collision_penalty,
                        state, True,
                    ))
                    if total_steps % 4 == 0:
                        self._update()
                    break

                reward = cfg.step_penalty + self._shaping(pose, nxt)
                done = self._is_goal(nxt)
                if done:
                    reward += cfg.goal_bonus
                    self.successful_episodes += 1

                next_state = self._state_features(nxt)
                self.replay.append((
                    state, a_idx, reward, next_state, done,
                ))
                if total_steps % 4 == 0:
                    self._update()

                if done:
                    break
                pose = nxt
                state = next_state

            eps = max(cfg.epsilon_end, eps - eps_step)
        self.training_time_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def rollout(
        self,
        start: Pose,
        goal: Pose,
        max_steps: int = 300,
        epsilon: float = 0.02,
    ) -> Tuple[List[Waypoint], bool, str]:
        self.goal = goal
        path: List[Waypoint] = [Waypoint(*start)]
        pose = start
        visited: Dict[Tuple[int, int, int], int] = {}

        for _ in range(max_steps):
            key = (int(round(pose[0] / 0.45)),
                   int(round(pose[1] / 0.45)),
                   int(round(pose[2] / 0.52)) % 12)
            visited[key] = visited.get(key, 0) + 1
            if visited[key] > 6:
                return path, False, "DQN: policy looping."

            state = self._state_features(pose)
            a_idx = self._select_action(state, epsilon)
            gear, steer = self.actions[a_idx]
            nxt = self._step(pose, gear, steer)

            if not self.grid.pose_is_valid(nxt):
                with torch.no_grad():
                    q = self.policy_net(state.unsqueeze(0)).squeeze(0)
                sorted_a = q.argsort(descending=True)
                found = False
                for alt in sorted_a:
                    alt = int(alt.item())
                    if alt == a_idx:
                        continue
                    g2, s2 = self.actions[alt]
                    nxt2 = self._step(pose, g2, s2)
                    if self.grid.pose_is_valid(nxt2):
                        nxt = nxt2
                        found = True
                        break
                if not found:
                    return path, False, "DQN: rollout hit obstacle."

            path.append(Waypoint(*nxt))
            pose = nxt
            if self._is_goal(pose):
                return path, True, "DQN: reached goal."

        return path, False, "DQN: exceeded step limit."


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def plan_dqn(
    pc: ParkingConfig,
    cc: CarConfig,
    cfg: Optional[DQNConfig] = None,
) -> TrajectoryResult:
    require_torch()
    lot = ParkingLot(pc, cc)
    obstacles = obstacles_for(lot)
    grid = OccupancyGrid(lot, obstacles=obstacles)

    if pc.parking_type == "perpendicular":
        goal = perpendicular_goal_pose(lot)
    elif pc.parking_type == "parallel":
        goal = parallel_goal_pose(lot)
    else:
        return TrajectoryResult(
            [], False, f"DQN: unknown parking type {pc.parking_type!r}.")

    planner = DQNPlanner(lot, cc, grid, cfg)
    t0 = time.perf_counter()
    planner.train(lot.car_start_pose, goal)
    train_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    path, ok, message = planner.rollout(lot.car_start_pose, goal)
    rollout_t = time.perf_counter() - t1

    final = path[-1] if path else Waypoint(*lot.car_start_pose)
    final_pos_err = math.hypot(final.x - goal[0], final.y - goal[1])
    final_head_err = abs(_angle_diff(final.theta, goal[2]))
    pl = _path_length(path)
    fully = grid.pose_is_fully_in_spot(
        (final.x, final.y, final.theta), margin=0.0)
    success = ok and fully

    if ok and not fully:
        message = "DQN: rollout reached goal but car not fully in spot."

    metrics = {
        "planning_time_s": train_t + rollout_t,
        "training_time_s": train_t,
        "rollout_time_s": rollout_t,
        "episodes": planner.episodes_run,
        "successful_episodes": planner.successful_episodes,
        "iterations": planner.episodes_run,
        "expanded_states": len(planner.replay),
        "path_length_m": pl,
        "waypoints": len(path),
        "final_pos_error_m": final_pos_err,
        "final_heading_error_deg": math.degrees(final_head_err),
        "fully_in_spot": fully,
        "obstacles": len(obstacles),
        "planner_kind": "dqn",
    }
    return TrajectoryResult(
        waypoints=path,
        feasible=success,
        message=message,
        phase_starts=[0],
        phase_names=["DQN rollout"],
        metrics=metrics,
    )
