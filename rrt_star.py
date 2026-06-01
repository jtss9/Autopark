"""
RRT* (Rapidly-exploring Random Tree Star) parking planner.

A sampling-based motion planner that builds a tree of kinematically
feasible trajectories and rewires the tree to improve path cost.
Unlike Hybrid A*, RRT* is probabilistically complete but not
deterministic — different runs may produce different paths.

Compared with Hybrid A* this planner:
  - explores more broadly (random sampling vs. grid expansion)
  - finds diverse solutions in obstacle-rich scenes
  - produces longer, less optimal paths on average
  - is slower to converge to the optimum (asymptotic optimality)

The planner uses the same OccupancyGrid, bicycle kinematics, and
goal-pose conventions as hybrid_astar.py so it slots into the
evaluator and simulator without changes.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from config import CarConfig, ParkingConfig
from geom import (
    angle_diff as _angle_diff,
    path_length as _path_length,
    wrap_pi as _wrap_pi,
)
from hybrid_astar import (
    OccupancyGrid,
    parallel_goal_pose,
    perpendicular_goal_pose,
)
import reeds_shepp
from parking_lot import ParkingLot, Rect
from scenarios import obstacles_for
from trajectory import TrajectoryResult, Waypoint

Pose = Tuple[float, float, float]


@dataclass
class RRTNode:
    x: float
    y: float
    theta: float
    cost: float = 0.0
    parent: int = -1
    trajectory: List[Pose] = field(default_factory=list)


class RRTStarPlanner:
    def __init__(
        self,
        lot: ParkingLot,
        cc: CarConfig,
        grid: OccupancyGrid,
    ):
        self.lot = lot
        self.cc = cc
        self.grid = grid

        self.step_size = 0.8
        self.integration_step = 0.20
        self.max_iterations = 20000
        self.goal_sample_rate = 0.20
        self.goal_xy_tol = 0.50
        self.goal_theta_tol = math.radians(15)
        self.rewire_radius = 2.5
        self.max_near = 12

        self.steering_set = (
            -cc.max_steer,
            -cc.max_steer * 0.5,
            0.0,
            cc.max_steer * 0.5,
            cc.max_steer,
        )

    def plan(
        self,
        start: Pose,
        goal: Pose,
        seed: int = 42,
    ) -> TrajectoryResult:
        rng = random.Random(seed)
        t0 = time.perf_counter()

        root = RRTNode(*start, cost=0.0, parent=-1)
        nodes: List[RRTNode] = [root]
        best_goal_idx: Optional[int] = None
        best_goal_cost = float("inf")

        for iteration in range(self.max_iterations):
            if time.perf_counter() - t0 > 60.0:
                break

            sample = self._sample(goal, rng)
            nearest_idx = self._nearest(nodes, sample)
            nearest = nodes[nearest_idx]

            new_pose, traj, steer = self._steer(
                (nearest.x, nearest.y, nearest.theta), sample, rng,
            )
            if new_pose is None:
                continue

            seg_cost = self._trajectory_cost(traj)
            new_cost = nearest.cost + seg_cost

            # Rewire: find a cheaper parent in the neighbourhood
            near_idxs = self._near(nodes, new_pose)
            best_parent = nearest_idx
            best_cost = new_cost
            best_traj = traj
            for ni in near_idxs:
                nn = nodes[ni]
                candidate_pose, candidate_traj, _ = self._steer(
                    (nn.x, nn.y, nn.theta), new_pose, rng,
                )
                if candidate_pose is None:
                    continue
                if math.hypot(candidate_pose[0] - new_pose[0],
                              candidate_pose[1] - new_pose[1]) > 0.5:
                    continue
                c = nn.cost + self._trajectory_cost(candidate_traj)
                if c < best_cost:
                    best_cost = c
                    best_parent = ni
                    best_traj = candidate_traj

            new_node = RRTNode(
                *new_pose,
                cost=best_cost,
                parent=best_parent,
                trajectory=best_traj,
            )
            new_idx = len(nodes)
            nodes.append(new_node)

            # Rewire neighbours through the new node
            for ni in near_idxs:
                nn = nodes[ni]
                rp, rt, _ = self._steer(new_pose, (nn.x, nn.y, nn.theta), rng)
                if rp is None:
                    continue
                if math.hypot(rp[0] - nn.x, rp[1] - nn.y) > 0.5:
                    continue
                rc = best_cost + self._trajectory_cost(rt)
                if rc < nn.cost:
                    nn.cost = rc
                    nn.parent = new_idx
                    nn.trajectory = rt

            if self._reached_goal(new_pose, goal) and best_cost < best_goal_cost:
                best_goal_idx = new_idx
                best_goal_cost = best_cost

            # Try an RS analytic shot from the new node
            if iteration % 5 == 0:
                dist_to_goal = math.hypot(
                    new_pose[0] - goal[0], new_pose[1] - goal[1])
                if dist_to_goal < 12.0:
                    shot = self._try_rs_shot(new_pose, goal)
                    if shot is not None:
                        shot_cost = best_cost + self._trajectory_cost(
                            [(w.x, w.y, w.theta) for w in shot])
                        if shot_cost < best_goal_cost:
                            shot_node = RRTNode(
                                *goal, cost=shot_cost, parent=new_idx,
                                trajectory=[(w.x, w.y, w.theta) for w in shot],
                            )
                            best_goal_idx = len(nodes)
                            best_goal_cost = shot_cost
                            nodes.append(shot_node)

        elapsed = time.perf_counter() - t0

        if best_goal_idx is None:
            return TrajectoryResult(
                [], False,
                f"RRT*: no path found in {iteration + 1} iterations.",
                metrics={
                    "planning_time_s": elapsed,
                    "iterations": iteration + 1,
                    "expanded_states": len(nodes),
                },
            )

        raw_path = self._reconstruct(best_goal_idx, nodes)
        smoothed = self._smooth_path(raw_path)
        final = smoothed[-1]
        final_pose = (final.x, final.y, final.theta)
        fully_in = self.grid.pose_is_fully_in_spot(final_pose, margin=0.0)

        metrics = {
            "planning_time_s": elapsed,
            "iterations": iteration + 1,
            "expanded_states": len(nodes),
            "path_length_m": _path_length(smoothed),
            "raw_path_length_m": _path_length(raw_path),
            "smoothed_path_length_m": _path_length(smoothed),
            "waypoints": len(smoothed),
            "raw_waypoints": len(raw_path),
            "smoothed_waypoints": len(smoothed),
            "final_pos_error_m": math.hypot(
                final.x - goal[0], final.y - goal[1]),
            "final_heading_error_deg": math.degrees(
                abs(_angle_diff(final.theta, goal[2]))),
            "fully_in_spot": fully_in,
            "obstacles": len(self.grid.obstacles),
        }
        msg = (f"RRT*: OK in {elapsed:.2f}s, "
               f"{iteration + 1} iterations, {len(nodes)} nodes.")
        return TrajectoryResult(
            waypoints=smoothed,
            feasible=fully_in,
            message=msg,
            phase_starts=[0],
            phase_names=["RRT* path"],
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample(self, goal: Pose, rng: random.Random) -> Pose:
        r = rng.random()
        if r < self.goal_sample_rate:
            return goal
        if r < self.goal_sample_rate + 0.10:
            spot = self.lot.spot_rect
            x = spot.x + rng.uniform(0, spot.w)
            y = spot.y + rng.uniform(-1, spot.h)
            theta = goal[2] + rng.uniform(-0.5, 0.5)
            return (x, y, theta)
        x = rng.uniform(self.grid.min_x + 1, self.grid.max_x - 1)
        y = rng.uniform(self.grid.min_y + 1, self.grid.max_y - 1)
        theta = rng.uniform(-math.pi, math.pi)
        return (x, y, theta)

    # ------------------------------------------------------------------
    # Nearest / near
    # ------------------------------------------------------------------
    def _pose_dist(self, a: Pose, b: Pose) -> float:
        return (math.hypot(a[0] - b[0], a[1] - b[1])
                + 0.5 * abs(_angle_diff(a[2], b[2])))

    def _nearest(self, nodes: List[RRTNode], pose: Pose) -> int:
        best_i, best_d = 0, float("inf")
        for i, n in enumerate(nodes):
            d = self._pose_dist((n.x, n.y, n.theta), pose)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def _near(self, nodes: List[RRTNode], pose: Pose) -> List[int]:
        r = self.rewire_radius
        near = [
            (math.hypot(n.x - pose[0], n.y - pose[1]), i)
            for i, n in enumerate(nodes)
            if math.hypot(n.x - pose[0], n.y - pose[1]) < r
        ]
        near.sort()
        return [i for _, i in near[: self.max_near]]

    # ------------------------------------------------------------------
    # Reeds-Shepp analytic shot
    # ------------------------------------------------------------------
    def _try_rs_shot(
        self, pose: Pose, goal: Pose,
    ) -> Optional[List[Waypoint]]:
        radius = self.cc.wheelbase / math.tan(self.cc.max_steer)
        rs_path = reeds_shepp.shortest_path(pose, goal, radius)
        if rs_path is None:
            return None
        dist = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        if rs_path.length > dist + 4.0 * radius:
            return None
        samples = reeds_shepp.discretize(pose, rs_path, radius, step=0.15)
        for x, y, theta, _ in samples:
            if not self.grid.pose_is_valid((x, y, theta)):
                return None
        final = samples[-1]
        if not self.grid.pose_is_fully_in_spot(
                (final[0], final[1], final[2])):
            return None
        return [Waypoint(x, y, theta) for x, y, theta, _ in samples]

    # ------------------------------------------------------------------
    # Steering (kinematic simulation)
    # ------------------------------------------------------------------
    def _steer(
        self,
        from_pose: Pose,
        to_pose: Pose,
        rng: random.Random,
    ) -> Tuple[Optional[Pose], List[Pose], float]:
        """Pick the best steering angle and simulate forward/reverse."""
        best_pose: Optional[Pose] = None
        best_traj: List[Pose] = []
        best_steer = 0.0
        best_score = float("inf")

        angle_to = math.atan2(
            to_pose[1] - from_pose[1],
            to_pose[0] - from_pose[0],
        )
        fwd_err = abs(_angle_diff(angle_to, from_pose[2]))
        gears = (-1, 1) if fwd_err > math.pi / 2 else (1, -1)

        for gear in gears:
            for steer in self.steering_set:
                pose, traj = self._simulate(from_pose, gear, steer)
                if pose is None:
                    continue
                d = self._pose_dist(pose, to_pose)
                if d < best_score:
                    best_score = d
                    best_pose = pose
                    best_traj = traj
                    best_steer = steer
            if best_pose is not None:
                break

        return best_pose, best_traj, best_steer

    def _simulate(
        self, start: Pose, gear: int, steer: float,
    ) -> Tuple[Optional[Pose], List[Pose]]:
        x, y, theta = start
        traj: List[Pose] = []
        travelled = 0.0
        while travelled < self.step_size:
            ds = min(self.integration_step, self.step_size - travelled)
            x += gear * ds * math.cos(theta)
            y += gear * ds * math.sin(theta)
            theta += gear * ds * math.tan(steer) / self.cc.wheelbase
            theta = _wrap_pi(theta)
            travelled += ds
            if not self.grid.pose_is_valid((x, y, theta)):
                if not traj:
                    return None, []
                return traj[-1], traj
            traj.append((x, y, theta))
        return (x, y, theta), traj

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def _trajectory_cost(self, traj: List[Pose]) -> float:
        if len(traj) < 2:
            return self.step_size
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(traj, traj[1:])
        )

    # ------------------------------------------------------------------
    # Goal / reconstruct / smooth
    # ------------------------------------------------------------------
    def _reached_goal(self, pose: Pose, goal: Pose) -> bool:
        if math.hypot(pose[0] - goal[0], pose[1] - goal[1]) > self.goal_xy_tol:
            return False
        if abs(_angle_diff(pose[2], goal[2])) > self.goal_theta_tol:
            return False
        return self.grid.pose_is_fully_in_spot(pose)

    def _reconstruct(self, idx: int, nodes: List[RRTNode]) -> List[Waypoint]:
        chain: List[int] = []
        while idx >= 0:
            chain.append(idx)
            idx = nodes[idx].parent
        chain.reverse()
        path: List[Waypoint] = [Waypoint(nodes[chain[0]].x,
                                         nodes[chain[0]].y,
                                         nodes[chain[0]].theta)]
        for ni in chain[1:]:
            for x, y, theta in nodes[ni].trajectory:
                path.append(Waypoint(x, y, theta))
        return path

    def _segment_valid(self, a: Waypoint, b: Waypoint) -> bool:
        dist = math.hypot(b.x - a.x, b.y - a.y)
        steps = max(2, int(math.ceil(dist / 0.12)))
        for i in range(steps + 1):
            t = i / steps
            x = a.x + (b.x - a.x) * t
            y = a.y + (b.y - a.y) * t
            theta = a.theta + _angle_diff(b.theta, a.theta) * t
            if not self.grid.pose_is_valid((x, y, theta)):
                return False
        return True

    def _smooth_path(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Remove redundant waypoints while preserving kinematic feasibility.

        A shortcut a→c is only taken when the heading change along the
        path segment AND the car-heading jump are both small, so the
        smoothed path never demands an instantaneous nonholonomic turn.
        """
        if len(waypoints) <= 2:
            return list(waypoints)
        cleaned = [waypoints[0]]
        for wp in waypoints[1:]:
            if math.hypot(wp.x - cleaned[-1].x,
                          wp.y - cleaned[-1].y) > 0.04:
                cleaned.append(wp)
        if len(cleaned) <= 2:
            return cleaned

        smoothed = [cleaned[0]]
        for i in range(1, len(cleaned) - 1):
            a = smoothed[-1]
            b = cleaned[i]
            c = cleaned[i + 1]
            ab = math.atan2(b.y - a.y, b.x - a.x)
            bc = math.atan2(c.y - b.y, c.x - b.x)
            heading_change = abs(_angle_diff(ab, bc))
            car_heading_change = abs(_angle_diff(c.theta, a.theta))
            if (heading_change < math.radians(7)
                    and car_heading_change < math.radians(10)
                    and self._segment_valid(a, c)):
                continue
            smoothed.append(b)
        smoothed.append(cleaned[-1])
        return smoothed


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def plan_rrt_star(
    pc: ParkingConfig,
    cc: CarConfig,
    obstacles: Optional[Sequence[Rect]] = None,
) -> TrajectoryResult:
    lot = ParkingLot(pc, cc)
    active_obstacles = (list(obstacles) if obstacles is not None
                        else obstacles_for(lot))
    grid = OccupancyGrid(lot, obstacles=active_obstacles)
    planner = RRTStarPlanner(lot, cc, grid)

    if pc.parking_type == "perpendicular":
        goal = perpendicular_goal_pose(lot)
    elif pc.parking_type == "parallel":
        goal = parallel_goal_pose(lot)
    else:
        return TrajectoryResult(
            [], False, f"RRT*: unknown parking type {pc.parking_type!r}.")

    result = planner.plan(lot.car_start_pose, goal)
    if result.phase_names:
        result.phase_names[0] = (
            "RRT* perpendicular parking"
            if pc.parking_type == "perpendicular"
            else "RRT* parallel parking"
        )
    return result
