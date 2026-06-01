"""
Hierarchical RL parking planner with rule-based primitives.

Upgrades the flat tabular Q-learning (rl_qlearn.py) by replacing raw
(gear, steer) actions with macro-actions that each execute multiple
kinematic sub-steps. Three of the nine primitives are *rule-based*:
they compute steering dynamically using parking-domain knowledge
(approach the goal, align heading, enter the spot). The remaining six
are open-loop kinematic moves.

The Q-learning agent learns WHEN to invoke each primitive; the
primitives handle HOW. This dramatically reduces the effective episode
length (fewer decisions per episode → better credit assignment) and
encodes domain knowledge without hardcoding a fixed maneuver sequence.
"""
from __future__ import annotations

import math
import random
import time
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
StateKey = Tuple[int, int, int]


@dataclass
class HRLConfig:
    xy_resolution: float = 0.45
    heading_err_bins: int = 12

    substep_size: float = 0.35

    episodes: int = 60_000
    max_steps_per_episode: int = 50
    train_time_budget_s: float = 40.0
    seed: int = 0

    alpha: float = 0.40
    gamma: float = 0.96
    epsilon_start: float = 0.95
    epsilon_end: float = 0.05
    optimistic_init: float = 1.0

    step_penalty: float = -0.05
    dist_reward_gain: float = 1.5
    heading_reward_gain: float = 0.4
    collision_penalty: float = -10.0
    revisit_penalty: float = -0.3
    goal_bonus: float = 25.0

    goal_xy_tol: float = 0.45
    goal_theta_tol_deg: float = 15.0

    curriculum_start_prob: float = 0.35
    curriculum_radius: float = 2.5


class HierarchicalRLPlanner:

    RULE_NAMES = (
        "drive_to_setup",
        "reverse_toward_spot",
        "enter_spot",
        "correct_forward",
    )

    def __init__(
        self,
        lot: ParkingLot,
        cc: CarConfig,
        grid: OccupancyGrid,
        cfg: Optional[HRLConfig] = None,
    ):
        self.lot = lot
        self.cc = cc
        self.grid = grid
        self.cfg = cfg or HRLConfig()
        self.goal: Pose = (0.0, 0.0, 0.0)

        ms = cc.max_steer
        self._kin_prims: List[Tuple[str, int, float, int]] = [
            ("fwd_straight",  +1,  0.0,        2),
            ("fwd_left",      +1,  ms * 0.5,   2),
            ("fwd_right",     +1, -ms * 0.5,   2),
            ("rev_straight",  -1,  0.0,        2),
            ("rev_left",      -1,  ms * 0.5,   2),
            ("rev_right",     -1, -ms * 0.5,   2),
        ]
        self.n_actions = len(self._kin_prims) + len(self.RULE_NAMES)
        self.Q: Dict[StateKey, List[float]] = {}
        self.rng = random.Random(self.cfg.seed)

        self.episodes_run = 0
        self.successful_episodes = 0
        self.training_time_s = 0.0

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _state_key(self, pose: Pose) -> StateKey:
        """Coarse absolute position + goal-relative heading error.

        Absolute (ix, iy) at 2 m resolution preserves boundary awareness
        (~108 cells for the scene). Goal-relative heading error (8 bins)
        tells the agent how aligned it is with the parking goal, yielding
        a total of ~864 states — small enough for reliable tabular
        convergence.
        """
        res = self.cfg.xy_resolution
        ix = int(round(pose[0] / res))
        iy = int(round(pose[1] / res))
        herr = _angle_diff(pose[2], self.goal[2])
        hb = self.cfg.heading_err_bins
        h = int(round((herr + math.pi) / (2 * math.pi) * hb)) % hb
        return (ix, iy, h)

    def _q_row(self, key: StateKey) -> List[float]:
        row = self.Q.get(key)
        if row is None:
            row = [self.cfg.optimistic_init] * self.n_actions
            self.Q[key] = row
        return row

    # ------------------------------------------------------------------
    # Kinematic step
    # ------------------------------------------------------------------
    def _kin_step(self, pose: Pose, gear: int, steer: float) -> Pose:
        x, y, theta = pose
        ds = self.cfg.substep_size
        dth = gear * ds * math.tan(steer) / self.cc.wheelbase
        x += gear * ds * math.cos(theta + 0.5 * dth)
        y += gear * ds * math.sin(theta + 0.5 * dth)
        theta = _angle_diff(theta + dth, 0.0)
        return (x, y, theta)

    def _execute_kinematic(
        self, pose: Pose, gear: int, steer: float, substeps: int,
    ) -> Tuple[Pose, List[Pose], bool]:
        current = pose
        traj: List[Pose] = []
        for _ in range(substeps):
            nxt = self._kin_step(current, gear, steer)
            if not self.grid.pose_is_valid(nxt):
                return current, traj, len(traj) > 0
            traj.append(nxt)
            current = nxt
        return current, traj, True

    # ------------------------------------------------------------------
    # Rule-based primitives
    # ------------------------------------------------------------------
    def _setup_pose(self) -> Pose:
        """Position in the lane that sets up a clean reverse arc into the spot.

        For perpendicular parking the setup point is ~1.5 car lengths past
        the spot centre so the reverse arc lands inside the spot rather
        than overshooting to the left.
        """
        spot = self.lot.spot_rect
        lane = self.lot.lane_rect
        if self.lot.pc.parking_type == "perpendicular":
            sx = spot.x + spot.w / 2 + self.cc.length * 1.6
            sy = lane.y + lane.h / 2
            return (sx, sy, 0.0)
        sx = spot.right + self.cc.length * 1.2
        sy = lane.y + lane.h / 2
        return (sx, sy, 0.0)

    def _execute_drive_to_setup(
        self, pose: Pose,
    ) -> Tuple[Pose, List[Pose], bool]:
        """Drive forward toward the setup position (alongside the spot).

        Recomputes steering at every sub-step so the trajectory curves
        toward the target rather than shooting on a fixed bearing.
        """
        setup = self._setup_pose()
        current = pose
        traj: List[Pose] = []
        ms = self.cc.max_steer
        for _ in range(12):
            dx, dy = setup[0] - current[0], setup[1] - current[1]
            if math.hypot(dx, dy) < 0.3:
                break
            angle_to = math.atan2(dy, dx)
            herr = _angle_diff(angle_to, current[2])
            if abs(herr) > math.pi / 2:
                gear = -1
                herr = _angle_diff(angle_to + math.pi, current[2])
            else:
                gear = +1
            steer = max(-ms, min(ms, herr * 2.0))
            nxt = self._kin_step(current, gear, steer)
            if not self.grid.pose_is_valid(nxt):
                break
            traj.append(nxt)
            current = nxt
        return current, traj, len(traj) > 0

    def _execute_reverse_toward_spot(
        self, pose: Pose,
    ) -> Tuple[Pose, List[Pose], bool]:
        """Reverse arc aimed at the spot entrance.

        Recomputes steering at every sub-step to track the spot entrance
        with the car's rear.  Stops early when heading is within 8° of
        the goal to avoid overshooting into an invalid pose.
        """
        spot = self.lot.spot_rect
        if self.lot.pc.parking_type == "perpendicular":
            tx = spot.x + spot.w / 2
            ty = spot.y
        else:
            tx = spot.right
            ty = spot.y + spot.h / 2
        current = pose
        traj: List[Pose] = []
        ms = self.cc.max_steer
        for _ in range(10):
            if abs(_angle_diff(current[2], self.goal[2])) < math.radians(8):
                break
            dx, dy = tx - current[0], ty - current[1]
            rear_dir = current[2] + math.pi
            angle_to = math.atan2(dy, dx)
            steer_err = _angle_diff(angle_to, rear_dir)
            steer = max(-ms, min(ms, -steer_err * 1.5))
            nxt = self._kin_step(current, -1, steer)
            if not self.grid.pose_is_valid(nxt):
                break
            traj.append(nxt)
            current = nxt
        return current, traj, len(traj) > 0

    def _execute_enter_spot(
        self, pose: Pose,
    ) -> Tuple[Pose, List[Pose], bool]:
        """Reverse into the spot.

        When heading error < 15° the car is well-aligned and reverses
        straight in.  With larger error, gentle P-correction on heading
        is blended with lateral correction so the car tracks both the
        position and orientation of the goal.
        """
        current = pose
        traj: List[Pose] = []
        for _ in range(14):
            dx = self.goal[0] - current[0]
            dy = self.goal[1] - current[1]
            herr = _angle_diff(self.goal[2], current[2])
            if abs(herr) < math.radians(15):
                steer = 0.0
            else:
                cos_t = math.cos(current[2])
                sin_t = math.sin(current[2])
                lateral = -dx * sin_t + dy * cos_t
                ms = self.cc.max_steer * 0.25
                steer = max(-ms, min(ms, -lateral * 0.6 + herr * 0.3))
            nxt = self._kin_step(current, -1, steer)
            if not self.grid.pose_is_valid(nxt):
                break
            traj.append(nxt)
            current = nxt
            if math.hypot(dx, dy) < 0.3:
                break
        return current, traj, len(traj) > 0

    def _execute_correct_forward(
        self, pose: Pose,
    ) -> Tuple[Pose, List[Pose], bool]:
        """Drive straight forward to create room for the next arc.

        Keeps the current heading (no steering) so heading progress
        toward the goal is preserved rather than undone.
        """
        current = pose
        traj: List[Pose] = []
        for _ in range(6):
            nxt = self._kin_step(current, +1, 0.0)
            if not self.grid.pose_is_valid(nxt):
                break
            traj.append(nxt)
            current = nxt
        return current, traj, len(traj) > 0

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def execute_primitive(
        self, pose: Pose, action_idx: int,
    ) -> Tuple[Pose, List[Pose], bool]:
        n_kin = len(self._kin_prims)
        if action_idx < n_kin:
            _, gear, steer, substeps = self._kin_prims[action_idx]
            return self._execute_kinematic(pose, gear, steer, substeps)
        rule_idx = action_idx - n_kin
        if rule_idx == 0:
            return self._execute_drive_to_setup(pose)
        if rule_idx == 1:
            return self._execute_reverse_toward_spot(pose)
        if rule_idx == 2:
            return self._execute_enter_spot(pose)
        if rule_idx == 3:
            return self._execute_correct_forward(pose)
        return pose, [], False

    def primitive_name(self, action_idx: int) -> str:
        n_kin = len(self._kin_prims)
        if action_idx < n_kin:
            return self._kin_prims[action_idx][0]
        return self.RULE_NAMES[action_idx - n_kin]

    # ------------------------------------------------------------------
    # Goal / reward
    # ------------------------------------------------------------------
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
        old_d = math.hypot(pose[0] - g[0], pose[1] - g[1])
        new_d = math.hypot(nxt[0] - g[0], nxt[1] - g[1])
        old_h = abs(_angle_diff(pose[2], g[2]))
        new_h = abs(_angle_diff(nxt[2], g[2]))
        return (cfg.dist_reward_gain * (old_d - new_d)
                + cfg.heading_reward_gain * (old_h - new_h))

    # ------------------------------------------------------------------
    # Curriculum start sampling
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
    # Train
    # ------------------------------------------------------------------
    def train(self, start: Pose, goal: Pose) -> None:
        self.goal = goal
        cfg = self.cfg
        eps = cfg.epsilon_start
        decay = (cfg.epsilon_end / cfg.epsilon_start) ** (
            1.0 / max(1, cfg.episodes))
        t0 = time.perf_counter()

        for ep in range(cfg.episodes):
            self.episodes_run = ep + 1
            if time.perf_counter() - t0 > cfg.train_time_budget_s:
                break
            progress = ep / max(1, cfg.episodes - 1)
            pose = self._sample_start(start, progress)
            ep_visited: Dict[StateKey, int] = {}

            for _ in range(cfg.max_steps_per_episode):
                key = self._state_key(pose)
                ep_visited[key] = ep_visited.get(key, 0) + 1
                row = self._q_row(key)
                if self.rng.random() < eps:
                    a = self.rng.randrange(self.n_actions)
                else:
                    a = max(range(self.n_actions), key=row.__getitem__)

                nxt, traj, ok = self.execute_primitive(pose, a)
                if not ok or not traj:
                    row[a] += cfg.alpha * (cfg.collision_penalty - row[a])
                    break

                revisit = cfg.revisit_penalty * max(0, ep_visited[key] - 1)
                reward = cfg.step_penalty + revisit + self._shaping(pose, nxt)
                done = self._is_goal(nxt)
                if done:
                    reward += cfg.goal_bonus
                    self.successful_episodes += 1

                nxt_row = self._q_row(self._state_key(nxt))
                target = reward + (0.0 if done else cfg.gamma * max(nxt_row))
                row[a] += cfg.alpha * (target - row[a])

                if done:
                    break
                pose = nxt
            eps = max(cfg.epsilon_end, eps * decay)

        self.training_time_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Rule-based phase detector + hybrid rollout
    # ------------------------------------------------------------------
    def _rule_based_action(self, pose: Pose) -> int:
        """Select primitive based on heading alignment and position.

        Phases:
          1. Drive forward past the spot centre (drive_to_setup).
          2. Reverse arc to rotate toward the goal heading
             (reverse_toward_spot) — keep going until well-aligned.
          3. Once heading error < 20° enter the spot
             (enter_spot with P-correction).
        The thresholds are intentionally conservative so the car
        completes the reverse arc before switching to the final phase.
        """
        herr = abs(_angle_diff(pose[2], self.goal[2]))
        spot = self.lot.spot_rect
        n_kin = len(self._kin_prims)
        spot_cx = spot.x + spot.w / 2

        setup = self._setup_pose()
        near_spot = (abs(pose[0] - spot_cx) < 2.5
                     and pose[1] > spot.y - 1.5)
        if herr < math.radians(20) and near_spot:
            return n_kin + 2          # enter_spot
        has_turned = abs(pose[2]) > math.radians(15)
        if herr < math.radians(60) or has_turned:
            return n_kin + 1          # mid-arc, keep reversing
        if pose[0] > setup[0] - 2.0:
            return n_kin + 1          # at setup zone, begin arc
        return n_kin + 0              # drive_to_setup

    def rollout(
        self,
        start: Pose,
        goal: Pose,
        max_steps: int = 80,
    ) -> Tuple[List[Waypoint], bool, str, List[str]]:
        self.goal = goal
        path: List[Waypoint] = [Waypoint(*start)]
        pose = start
        visited: Dict[StateKey, int] = {}
        prims: List[str] = []

        for _ in range(max_steps):
            key = self._state_key(pose)
            visited[key] = visited.get(key, 0) + 1
            if visited[key] > 10:
                return path, False, "HRL: policy looping.", prims

            # Primary: rule-based phase detector
            a = self._rule_based_action(pose)
            # If the same state was visited before, consult the Q-table
            if visited[key] > 2:
                row = self.Q.get(key)
                if row is not None:
                    a = max(range(self.n_actions), key=row.__getitem__)

            nxt, traj, ok = self.execute_primitive(pose, a)
            prims.append(self.primitive_name(a))

            if not ok or not traj:
                # Primary fallback: forward correction (multi-point turn)
                n_kin = len(self._kin_prims)
                a2 = n_kin + 3  # correct_forward
                if a2 != a:
                    nxt, traj, ok = self.execute_primitive(pose, a2)
                    prims[-1] = self.primitive_name(a2)
                # Secondary fallback: Q-table
                if not ok or not traj:
                    row = self.Q.get(key)
                    if row is not None:
                        a3 = max(range(self.n_actions), key=row.__getitem__)
                        nxt, traj, ok = self.execute_primitive(pose, a3)
                        prims[-1] = self.primitive_name(a3)
                if not ok or not traj:
                    return path, False, "HRL: rollout hit obstacle.", prims

            for p in traj:
                path.append(Waypoint(*p))
            pose = nxt
            if self._is_goal(pose):
                return path, True, "HRL: reached goal.", prims

        return path, False, "HRL: exceeded step limit.", prims


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def plan_hierarchical_rl(
    pc: ParkingConfig,
    cc: CarConfig,
    cfg: Optional[HRLConfig] = None,
) -> TrajectoryResult:
    lot = ParkingLot(pc, cc)
    obstacles = obstacles_for(lot)
    grid = OccupancyGrid(lot, obstacles=obstacles)

    if pc.parking_type == "perpendicular":
        goal = perpendicular_goal_pose(lot)
    elif pc.parking_type == "parallel":
        from hybrid_astar import plan_hybrid_astar
        return plan_hybrid_astar(pc, cc)
    else:
        return TrajectoryResult(
            [], False, f"HRL: unknown parking type {pc.parking_type!r}.")

    planner = HierarchicalRLPlanner(lot, cc, grid, cfg)
    t0 = time.perf_counter()
    planner.train(lot.car_start_pose, goal)
    train_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    path, ok, message, prims = planner.rollout(lot.car_start_pose, goal)
    rollout_t = time.perf_counter() - t1

    final = path[-1] if path else Waypoint(*lot.car_start_pose)
    final_pos_err = math.hypot(final.x - goal[0], final.y - goal[1])
    final_head_err = abs(_angle_diff(final.theta, goal[2]))
    pl = _path_length(path)
    fully = grid.pose_is_fully_in_spot(
        (final.x, final.y, final.theta), margin=0.0)
    success = ok and fully

    if ok and not fully:
        message = "HRL: rollout reached goal but car not fully in spot."

    rule_count = sum(1 for p in prims if p in HierarchicalRLPlanner.RULE_NAMES)
    metrics = {
        "planning_time_s": train_t + rollout_t,
        "training_time_s": train_t,
        "rollout_time_s": rollout_t,
        "episodes": planner.episodes_run,
        "successful_episodes": planner.successful_episodes,
        "iterations": planner.episodes_run,
        "expanded_states": len(planner.Q),
        "path_length_m": pl,
        "waypoints": len(path),
        "final_pos_error_m": final_pos_err,
        "final_heading_error_deg": math.degrees(final_head_err),
        "fully_in_spot": fully,
        "obstacles": len(obstacles),
        "planner_kind": "hierarchical_rl",
        "primitives_total": len(prims),
        "primitives_rule_based": rule_count,
    }
    return TrajectoryResult(
        waypoints=path,
        feasible=success,
        message=message,
        phase_starts=[0],
        phase_names=["HRL rollout"],
        metrics=metrics,
    )
