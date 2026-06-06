"""
Hybrid A* parking planner.

This module is the first step toward replacing the fixed parking arc with a
general planner over SE(2): x, y, and heading. It returns the same
TrajectoryResult/Waypoint objects used by the current simulator.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff
from parking_lot import ParkingLot, Rect
import reeds_shepp
from scenarios import obstacles_for
from trajectory import TrajectoryResult, Waypoint


Pose = Tuple[float, float, float]


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return (1.0, 0.0)
    return (dx / n, dy / n)


def _separated_on_axis(poly_a, poly_b, axis) -> bool:
    """True if the two convex polygons' projections onto `axis` do not overlap."""
    ax, ay = axis
    a_proj = [px * ax + py * ay for px, py in poly_a]
    b_proj = [px * ax + py * ay for px, py in poly_b]
    return max(a_proj) < min(b_proj) or max(b_proj) < min(a_proj)


def _obb_hits_rect(car_corners, rect: "Rect", margin: float = 0.0) -> bool:
    """Exact overlap test between the oriented car footprint (4 corners) and an
    axis-aligned obstacle Rect, via the Separating Axis Theorem.

    Only four candidate axes are needed: world x, world y, and the car's two
    edge directions. If any axis separates the two polygons, they don't touch.
    """
    rx0, ry0 = rect.x - margin, rect.y - margin
    rx1, ry1 = rect.right + margin, rect.top + margin
    rect_pts = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]
    (ax, ay), (bx, by), _c, (dx, dy) = car_corners  # rear-left, front-left, _, rear-right
    axes = [
        (1.0, 0.0),
        (0.0, 1.0),
        _unit(bx - ax, by - ay),
        _unit(dx - ax, dy - ay),
    ]
    for axis in axes:
        if _separated_on_axis(car_corners, rect_pts, axis):
            return False
    return True


@dataclass(frozen=True)
class GridIndex:
    ix: int
    iy: int
    ith: int


@dataclass
class SearchNode:
    x: float
    y: float
    theta: float
    g: float
    parent: Optional[GridIndex]
    direction: int
    steer: float


class OccupancyGrid:
    """
    Lightweight occupancy grid backed by ParkingLot geometry.

    The grid stores static obstacle cells, while validity still uses continuous
    car-corner checks against the lane/spot shape so the car body is not reduced
    to a point.
    """

    def __init__(
        self,
        lot: ParkingLot,
        resolution: float = 0.25,
        obstacles: Optional[Sequence[Rect]] = None,
    ):
        self.lot = lot
        self.resolution = resolution
        self.obstacles = list(obstacles or [])
        self.obstacle_margin = 0.05   # safety clearance for the SAT body check

        min_x = min(lot.lane_rect.x, lot.spot_rect.x)
        min_y = min(lot.lane_rect.y, lot.spot_rect.y)
        max_x = max(lot.lane_rect.right, lot.spot_rect.right)
        max_y = max(lot.lane_rect.top, lot.spot_rect.top)
        pad = max(lot.cc.length, lot.cc.width)

        self.min_x = min_x - pad
        self.min_y = min_y - pad
        self.max_x = max_x + pad
        self.max_y = max_y + pad
        self.width = int(math.ceil((self.max_x - self.min_x) / resolution))
        self.height = int(math.ceil((self.max_y - self.min_y) / resolution))

        self.blocked = set()
        for obs in self.obstacles:
            self._rasterize_obstacle(obs)

    def _rasterize_obstacle(self, rect: Rect) -> None:
        ix0, iy0 = self.world_to_cell(rect.x, rect.y)
        ix1, iy1 = self.world_to_cell(rect.right, rect.top)
        for ix in range(min(ix0, ix1), max(ix0, ix1) + 1):
            for iy in range(min(iy0, iy1), max(iy0, iy1) + 1):
                self.blocked.add((ix, iy))

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(math.floor((x - self.min_x) / self.resolution))
        iy = int(math.floor((y - self.min_y) / self.resolution))
        return ix, iy

    def contains_point(self, x: float, y: float) -> bool:
        ix, iy = self.world_to_cell(x, y)
        return 0 <= ix < self.width and 0 <= iy < self.height

    def point_is_blocked(self, x: float, y: float) -> bool:
        ix, iy = self.world_to_cell(x, y)
        return (ix, iy) in self.blocked

    def _point_in_rect(self, x: float, y: float, rect: Rect, margin: float) -> bool:
        return (
            rect.x - margin <= x <= rect.right + margin
            and rect.y - margin <= y <= rect.top + margin
        )

    def pose_is_fully_in_spot(self, pose: Pose, margin: float = 0.02) -> bool:
        spot = self.lot.spot_rect
        return all(
            self._point_in_rect(cx, cy, spot, margin)
            for cx, cy in self.lot.car_corners(pose)
        )

    def pose_is_valid(self, pose: Pose, margin: float = 0.03) -> bool:
        x, y, theta = pose
        if not self.contains_point(x, y):
            return False

        lane = self.lot.lane_rect
        spot = self.lot.spot_rect
        corners = self.lot.car_corners((x, y, theta))
        for cx, cy in corners:
            in_lane = self._point_in_rect(cx, cy, lane, margin)
            in_spot = self._point_in_rect(cx, cy, spot, margin)
            if not (in_lane or in_spot):
                return False
            if self.point_is_blocked(cx, cy):
                return False
        # Exact car-body vs obstacle test (SAT) — catches small obstacles that
        # sit under a car edge with no corner landing in a blocked cell.
        for obs in self.obstacles:
            if _obb_hits_rect(corners, obs, self.obstacle_margin):
                return False
        return True


class HybridAStarPlanner:
    def __init__(
        self,
        lot: ParkingLot,
        car_config: CarConfig,
        grid: Optional[OccupancyGrid] = None,
    ):
        self.lot = lot
        self.cc = car_config
        self.grid = grid or OccupancyGrid(lot)

        self.xy_resolution = 0.10
        self.theta_bins = 36
        self.motion_step = 0.45
        self.integration_step = 0.09
        self.goal_xy_tolerance = 0.35
        self.goal_theta_tolerance = math.radians(12)
        self.max_iterations = 150000

        self.steering_set = (
            -self.cc.max_steer,
            -self.cc.max_steer * 0.5,
            0.0,
            self.cc.max_steer * 0.5,
            self.cc.max_steer,
        )

        # Reeds-Shepp analytic shot: try once every N expansions and
        # always when the node is close enough to the goal.
        self.rs_radius = max(self.cc.min_turn_radius, 0.5)
        self.rs_step = 0.12
        self.rs_shot_interval = 50
        self.rs_shot_radius = 4.5  # try unconditionally inside this distance

        # Diagnostic counters populated by plan()
        self.rs_shot_attempts = 0
        self.rs_shot_successes = 0

        # Cache RS-heuristic values on a coarse (xy,theta) grid to avoid
        # recomputing the same RS path for every popped node.
        self._heuristic_cache: Dict[Tuple[int, int, int], float] = {}
        self._heuristic_xy = 0.3
        self._heuristic_theta_bins = 24

    def plan(self, start: Pose, goal: Pose) -> TrajectoryResult:
        start_t = time.perf_counter()
        self.rs_shot_attempts = 0
        self.rs_shot_successes = 0
        self._heuristic_cache.clear()
        if not self.grid.pose_is_valid(start):
            return TrajectoryResult([], False, "Hybrid A*: start pose is invalid.")
        if not self.grid.pose_is_valid(goal):
            return TrajectoryResult([], False, "Hybrid A*: goal pose is invalid.")

        open_heap: List[Tuple[float, int, GridIndex]] = []
        nodes: Dict[GridIndex, SearchNode] = {}
        best_cost: Dict[GridIndex, float] = {}
        closed = set()

        start_idx = self._index(start)
        start_node = SearchNode(*start, g=0.0, parent=None, direction=0, steer=0.0)
        nodes[start_idx] = start_node
        best_cost[start_idx] = 0.0
        counter = 0
        heapq.heappush(open_heap, (self._heuristic(start, goal), counter, start_idx))

        iterations = 0
        while open_heap and iterations < self.max_iterations:
            iterations += 1
            _, _, current_idx = heapq.heappop(open_heap)
            if current_idx in closed:
                continue
            closed.add(current_idx)
            current = nodes[current_idx]
            current_pose: Pose = (current.x, current.y, current.theta)

            if self._reached_goal(current_pose, goal):
                raw_wps = self._reconstruct(current_idx, nodes)
                wps = self._smooth_path(raw_wps)
                elapsed = time.perf_counter() - start_t
                metrics = self._metrics(
                    wps,
                    goal,
                    elapsed,
                    iterations,
                    len(nodes),
                    raw_wps,
                )
                return TrajectoryResult(
                    wps,
                    True,
                    f"Hybrid A*: OK in {elapsed:.2f}s, {iterations} iterations.",
                    [0],
                    ["Hybrid A* parking"],
                    metrics,
                )

            shot = self._try_rs_shot(current_pose, goal, iterations)
            if shot is not None:
                base_wps = self._reconstruct(current_idx, nodes)
                merged = self._merge_shot(base_wps, shot)
                wps = self._smooth_path(merged)
                elapsed = time.perf_counter() - start_t
                metrics = self._metrics(
                    wps,
                    goal,
                    elapsed,
                    iterations,
                    len(nodes),
                    merged,
                )
                metrics["rs_shot_attempts"] = self.rs_shot_attempts
                metrics["rs_shot_successes"] = self.rs_shot_successes
                metrics["used_analytic_shot"] = True
                return TrajectoryResult(
                    wps,
                    True,
                    f"Hybrid A*+RS: OK in {elapsed:.2f}s, {iterations} iterations "
                    f"(analytic shot).",
                    [0],
                    ["Hybrid A* + Reeds-Shepp parking"],
                    metrics,
                )

            for pose, cost, direction, steer in self._expand(current):
                idx = self._index(pose)
                new_g = current.g + cost
                if idx in closed:
                    continue
                if new_g >= best_cost.get(idx, float("inf")):
                    continue

                best_cost[idx] = new_g
                nodes[idx] = SearchNode(
                    pose[0],
                    pose[1],
                    pose[2],
                    new_g,
                    current_idx,
                    direction,
                    steer,
                )
                counter += 1
                f_score = new_g + self._heuristic(pose, goal)
                heapq.heappush(open_heap, (f_score, counter, idx))

        return TrajectoryResult(
            [],
            False,
            f"Hybrid A*: no path found within {self.max_iterations} iterations.",
            [],
            [],
        )

    def _expand(self, node: SearchNode) -> Iterable[Tuple[Pose, float, int, float]]:
        for direction in (1, -1):
            for steer in self.steering_set:
                pose = self._simulate((node.x, node.y, node.theta), direction, steer)
                if pose is None:
                    continue

                cost = self.motion_step
                if direction < 0:
                    cost *= 1.15
                if node.direction and direction != node.direction:
                    cost += 1.0
                cost += 0.08 * abs(steer)
                cost += 0.25 * abs(steer - node.steer)
                yield pose, cost, direction, steer

    def _simulate(self, start: Pose, direction: int, steer: float) -> Optional[Pose]:
        x, y, theta = start
        travelled = 0.0
        while travelled < self.motion_step:
            ds = min(self.integration_step, self.motion_step - travelled)
            x += direction * ds * math.cos(theta)
            y += direction * ds * math.sin(theta)
            theta += direction * ds * math.tan(steer) / self.cc.wheelbase
            theta = (theta + math.pi) % (2 * math.pi) - math.pi
            travelled += ds
            if not self.grid.pose_is_valid((x, y, theta)):
                return None
        return x, y, theta

    def _index(self, pose: Pose) -> GridIndex:
        x, y, theta = pose
        ix = int(round(x / self.xy_resolution))
        iy = int(round(y / self.xy_resolution))
        ith = int(round(((theta + math.pi) % (2 * math.pi)) / (2 * math.pi) * self.theta_bins))
        ith %= self.theta_bins
        return GridIndex(ix, iy, ith)

    def _heuristic(self, pose: Pose, goal: Pose) -> float:
        dx = pose[0] - goal[0]
        dy = pose[1] - goal[1]
        dist = math.hypot(dx, dy)
        dtheta = abs(_angle_diff(pose[2], goal[2]))
        holonomic = dist + 0.4 * dtheta
        # Reeds-Shepp gives a non-holonomic lower bound but is expensive.
        # Use a cached lookup on a coarse grid so each pose costs O(1)
        # after the first RS query in its bucket.
        if dist < self.rs_shot_radius * 1.5:
            key = (
                int(round((pose[0] - goal[0]) / self._heuristic_xy)),
                int(round((pose[1] - goal[1]) / self._heuristic_xy)),
                int(round(
                    _angle_diff(pose[2], goal[2])
                    / (2 * math.pi)
                    * self._heuristic_theta_bins
                )),
            )
            rs_len = self._heuristic_cache.get(key)
            if rs_len is None:
                rs_len = reeds_shepp.path_length(pose, goal, self.rs_radius)
                self._heuristic_cache[key] = rs_len
            if math.isfinite(rs_len):
                return max(holonomic, rs_len)
        return holonomic

    def _try_rs_shot(
        self,
        pose: Pose,
        goal: Pose,
        iterations: int,
    ) -> Optional[List[Waypoint]]:
        dist = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        if dist > self.rs_shot_radius:
            if iterations % self.rs_shot_interval != 0:
                return None
            if dist > self.rs_shot_radius * 2.5:
                return None

        self.rs_shot_attempts += 1
        rs_path = reeds_shepp.shortest_path(pose, goal, self.rs_radius)
        if rs_path is None:
            return None
        # Cap shot length to avoid validating very long segments far from goal
        if rs_path.length > dist + 4.0 * self.rs_radius:
            return None

        samples = reeds_shepp.discretize(pose, rs_path, self.rs_radius, step=self.rs_step)
        for x, y, theta, _ in samples:
            if not self.grid.pose_is_valid((x, y, theta)):
                return None

        # Final pose must satisfy the same parking acceptance criteria
        final = samples[-1]
        if not self.grid.pose_is_fully_in_spot((final[0], final[1], final[2])):
            return None

        self.rs_shot_successes += 1
        return [Waypoint(x, y, theta) for x, y, theta, _ in samples]

    def _merge_shot(
        self,
        base_path: List[Waypoint],
        shot: List[Waypoint],
    ) -> List[Waypoint]:
        if not base_path:
            return list(shot)
        # The shot starts at the last base waypoint; skip its first sample.
        return list(base_path) + list(shot[1:])

    def _reached_goal(self, pose: Pose, goal: Pose) -> bool:
        dist = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        heading_err = abs(_angle_diff(pose[2], goal[2]))
        return (
            dist <= self.goal_xy_tolerance
            and heading_err <= self.goal_theta_tolerance
            and self.grid.pose_is_fully_in_spot(pose)
        )

    def _reconstruct(
        self,
        goal_idx: GridIndex,
        nodes: Dict[GridIndex, SearchNode],
    ) -> List[Waypoint]:
        path = []
        idx: Optional[GridIndex] = goal_idx
        while idx is not None:
            node = nodes[idx]
            path.append(Waypoint(node.x, node.y, node.theta))
            idx = node.parent
        path.reverse()
        return path

    def _segment_is_valid(self, a: Waypoint, b: Waypoint) -> bool:
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

    def _smooth_path(self, waypoints: Sequence[Waypoint]) -> List[Waypoint]:
        if len(waypoints) <= 2:
            return list(waypoints)

        cleaned = [waypoints[0]]
        for wp in waypoints[1:]:
            prev = cleaned[-1]
            if math.hypot(wp.x - prev.x, wp.y - prev.y) < 0.05:
                continue
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
            if (
                heading_change < math.radians(7)
                and car_heading_change < math.radians(10)
                and self._segment_is_valid(a, c)
            ):
                continue
            smoothed.append(b)
        smoothed.append(cleaned[-1])
        return smoothed

    def _path_length(self, waypoints: Sequence[Waypoint]) -> float:
        return sum(
            math.hypot(b.x - a.x, b.y - a.y)
            for a, b in zip(waypoints, waypoints[1:])
        )

    def _metrics(
        self,
        waypoints: Sequence[Waypoint],
        goal: Pose,
        planning_time_s: float,
        iterations: int,
        expanded_states: int,
        raw_waypoints: Optional[Sequence[Waypoint]] = None,
    ) -> dict:
        raw_waypoints = raw_waypoints or waypoints
        path_length = self._path_length(waypoints)
        raw_path_length = self._path_length(raw_waypoints)

        final = waypoints[-1]
        final_pose = (final.x, final.y, final.theta)
        final_pos_error = math.hypot(final.x - goal[0], final.y - goal[1])
        final_heading_error = abs(_angle_diff(final.theta, goal[2]))

        return {
            "planning_time_s": planning_time_s,
            "iterations": iterations,
            "expanded_states": expanded_states,
            "path_length_m": path_length,
            "raw_path_length_m": raw_path_length,
            "smoothed_path_length_m": path_length,
            "waypoints": len(waypoints),
            "raw_waypoints": len(raw_waypoints),
            "smoothed_waypoints": len(waypoints),
            "final_pos_error_m": final_pos_error,
            "final_heading_error_deg": math.degrees(final_heading_error),
            "fully_in_spot": self.grid.pose_is_fully_in_spot(final_pose, margin=0.0),
            "obstacles": len(self.grid.obstacles),
        }


def perpendicular_goal_pose(lot: ParkingLot) -> Pose:
    """Return a rear-axle goal pose centered inside a perpendicular spot."""
    spot = lot.spot_rect
    x = spot.x + spot.w / 2
    y = spot.top - 0.15
    theta = -math.pi / 2
    return x, y, theta


def parallel_goal_pose(lot: ParkingLot) -> Pose:
    """Return a rear-axle goal pose inside a parallel spot."""
    spot = lot.spot_rect
    x = spot.x + 0.15
    y = spot.y + spot.h / 2
    theta = 0.0
    return x, y, theta


def _obstacle_rects(pc: ParkingConfig, lot: ParkingLot) -> List[Rect]:
    """Scenario obstacles plus the user-placed obstacle (if any)."""
    rects = list(obstacles_for(lot))
    if pc.obstacle is not None:
        x, y, w, h = pc.obstacle
        rects.append(Rect(x, y, w, h))
    return rects


def plan_hybrid_astar(
    pc: ParkingConfig,
    cc: CarConfig,
    obstacles: Optional[Sequence[Rect]] = None,
) -> TrajectoryResult:
    lot = ParkingLot(pc, cc)
    active_obstacles = list(obstacles) if obstacles is not None else _obstacle_rects(pc, lot)
    grid = OccupancyGrid(lot, obstacles=active_obstacles)
    planner = HybridAStarPlanner(lot, cc, grid)
    if pc.parking_type == "perpendicular":
        goal = perpendicular_goal_pose(lot)
    elif pc.parking_type == "parallel":
        goal = parallel_goal_pose(lot)
    else:
        return TrajectoryResult([], False, f"Hybrid A*: unknown parking type {pc.parking_type!r}.")

    print("Planning Hybrid A* trajectory...", end="", flush=True)
    t0 = time.perf_counter()
    result = planner.plan(lot.car_start_pose, goal)
    elapsed = time.perf_counter() - t0
    if result.feasible:
        print(f" done in {elapsed:.1f}s ({len(result.waypoints)} waypoints)")
    else:
        print(f" failed ({elapsed:.1f}s): {result.message}")
    if result.phase_names:
        result.phase_names[0] = (
            "Hybrid A* perpendicular parking"
            if pc.parking_type == "perpendicular"
            else "Hybrid A* parallel parking"
        )
    return result
