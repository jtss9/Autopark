"""
Gymnasium environment for RL-based parking trajectory optimization.

Wraps existing codebase primitives (ParkingLot, CarDynamics, OccupancyGrid)
into a standard Gymnasium interface with continuous observation and action spaces.

Supports both flat observation (for vanilla SAC) and goal-conditioned dict
observation (for HER).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "RL environment requires gymnasium. Install with: "
        "pip install -r requirements-rl.txt"
    ) from e

from config import CarConfig, ParkingConfig
from controller import CarDynamics
from geom import angle_diff, wrap_pi
from hybrid_astar import (
    HybridAStarPlanner,
    OccupancyGrid,
    perpendicular_goal_pose,
    parallel_goal_pose,
)
from parking_lot import ParkingLot, Rect
from scenarios import obstacles_for
import reeds_shepp


Pose = Tuple[float, float, float]


@dataclass
class CurriculumStage:
    """Reverse-curriculum stage.

    Starts are sampled along the hybrid A* reference path (from the lot's
    fixed start pose to the goal): ``path_t_lo`` is the lowest path fraction
    a start may be placed at (1.0 = at the goal, 0.0 = the fixed start), so
    early stages begin near the goal and later stages cover the whole
    maneuver. ``fixed_start_ratio`` of episodes start at the exact
    ``ParkingLot.car_start_pose`` (small noise) — the pose `python main.py`
    uses. ``lane_random_ratio`` adds random in-lane approach poses for
    robustness.
    """
    name: str
    path_t_lo: float
    fixed_start_ratio: float
    lane_random_ratio: float = 0.0
    obstacle_scenario: str = "none"
    success_rate_threshold: float = 0.6
    # Legacy fields (kept so older callers/prints continue to work).
    max_start_distance: float = 11.0
    max_start_heading_error: float = math.pi


CURRICULUM_STAGES = [
    # Stage 0: starts in/near the spot entry — learn the final reverse-in.
    CurriculumStage("spot_entry", path_t_lo=0.60, fixed_start_ratio=0.10,
                    success_rate_threshold=0.70),
    # Stage 1: starts around the gear-switch point past the spot.
    CurriculumStage("mid_path", path_t_lo=0.30, fixed_start_ratio=0.20,
                    success_rate_threshold=0.60),
    # Stage 2: full maneuver from the fixed start.
    CurriculumStage("full_path", path_t_lo=0.0, fixed_start_ratio=0.50,
                    success_rate_threshold=0.60),
    # Stage 3: add random in-lane approach poses for robustness.
    CurriculumStage("robust", path_t_lo=0.0, fixed_start_ratio=0.40,
                    lane_random_ratio=0.30, success_rate_threshold=0.50),
    # Obstacle stages can be appended here once the no-obstacle task is solved.
]


# Reference paths are expensive (hybrid A*); cache per world configuration.
_REF_PATH_CACHE: Dict[Tuple, Optional[List[Pose]]] = {}


def _compute_reference_path(
    lot: ParkingLot, cc: CarConfig, grid: OccupancyGrid, goal: Pose,
) -> Optional[List[Pose]]:
    """Hybrid A* path from the lot's fixed start to the goal (or None).

    Planned with an inflated car body and reduced max steering so the
    reference keeps clearance and curvature margin: raw A* paths hug the
    boundary with ~3 cm clearance at min turn radius, which no tracker (or
    policy) can follow without colliding. Falls back to tighter settings if
    the conservative plan is infeasible.
    """
    pc = lot.pc
    key = (
        pc.parking_type, pc.obstacle_scenario,
        round(pc.lane_width, 3), round(pc.spot_length, 3),
        round(pc.spot_width, 3),
        round(cc.length, 3), round(cc.width, 3),
        pc.obstacle,
    )
    if key in _REF_PATH_CACHE:
        return _REF_PATH_CACHE[key]

    path = None
    for inflate_l, inflate_w, steer_scale in (
        (0.50, 0.30, 0.80),
        (0.35, 0.20, 0.85),
        (0.15, 0.10, 0.90),
        (0.0, 0.0, 1.0),
    ):
        plan_cc = CarConfig(length=cc.length + inflate_l, width=cc.width + inflate_w)
        plan_cc.max_steer = cc.max_steer * steer_scale
        plan_lot = ParkingLot(pc, plan_cc)
        plan_grid = OccupancyGrid(plan_lot, obstacles=grid.obstacles)
        planner = HybridAStarPlanner(plan_lot, plan_cc, plan_grid)
        result = planner.plan(lot.car_start_pose, goal)
        if result.feasible and len(result.waypoints) >= 2:
            path = [(w.x, w.y, w.theta) for w in result.waypoints]
            break

    _REF_PATH_CACHE[key] = path
    return path


class ParkingEnv(gym.Env):
    """Continuous-control parking environment using the bicycle kinematic model.

    Observation: flat vector of relative pose, velocity, steering, obstacle features.
    Action: [steering_cmd, speed_cmd] in [-1, 1].
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    # Max number of nearest obstacles encoded in observation
    MAX_OBS_K = 4

    def __init__(
        self,
        parking_config: Optional[ParkingConfig] = None,
        car_config: Optional[CarConfig] = None,
        curriculum_stage: int = 0,
        max_episode_steps: int = 500,
        # dt 0.2 (not 0.1): with tiny steps the one-step consequence of an
        # action is ~2% of the value scale, leaving the critic almost flat
        # across actions — policies froze because Q(stop) ~= Q(maneuver).
        dt: float = 0.2,
        max_speed: float = 1.5,
        goal_conditioned: bool = False,
        reward_type: str = "dense",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        fixed_start_ratio: float = 0.6,
        proximity_margin: float = 0.3,
        proximity_penalty_scale: float = 1.0,
    ):
        super().__init__()

        self.pc = parking_config or ParkingConfig()
        self.cc = car_config or CarConfig()
        self.dt = dt
        self.max_speed = max_speed
        self.max_episode_steps = max_episode_steps
        self.goal_conditioned = goal_conditioned
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.fixed_start_ratio = fixed_start_ratio
        self.proximity_margin = proximity_margin
        self.proximity_penalty_scale = proximity_penalty_scale

        self._curriculum_stage_idx = min(
            curriculum_stage, len(CURRICULUM_STAGES) - 1
        )
        self._rng = np.random.default_rng(seed)

        # Build world geometry
        self._rebuild_world()

        # Action space: [steering_cmd, speed_cmd] each in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation space
        obs_dim = self._obs_dim()
        if self.goal_conditioned:
            goal_dim = 4  # x, y, cos(theta), sin(theta)
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
            })
        else:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (obs_dim,), np.float32
            )

        # Episode state
        self._car: Optional[CarDynamics] = None
        self._velocity: float = 0.0
        self._steering: float = 0.0
        self._prev_steering: float = 0.0
        self._step_count: int = 0
        self._prev_pos_error: float = 0.0
        self._prev_heading_error: float = 0.0
        self._gear_switches: int = 0
        self._prev_velocity_sign: int = 0

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _rebuild_world(self) -> None:
        stage = CURRICULUM_STAGES[self._curriculum_stage_idx]
        pc = ParkingConfig(
            lane_width=self.pc.lane_width,
            spot_length=self.pc.spot_length,
            spot_width=self.pc.spot_width,
            parking_type=self.pc.parking_type,
            obstacle_scenario=stage.obstacle_scenario,
            planner=self.pc.planner,
            obstacle=self.pc.obstacle,
        )
        self._active_pc = pc
        self.lot = ParkingLot(pc, self.cc)
        obstacles = obstacles_for(self.lot)
        if pc.obstacle is not None:
            x, y, w, h = pc.obstacle
            obstacles.append(Rect(x, y, w, h))
        self._obstacles = obstacles
        self.grid = OccupancyGrid(self.lot, obstacles=obstacles)

        if pc.parking_type == "perpendicular":
            self._goal_pose = perpendicular_goal_pose(self.lot)
        else:
            self._goal_pose = parallel_goal_pose(self.lot)

        # Reeds-Shepp radius for the distance-to-go potential (matches the
        # hybrid A* heuristic radius).
        self._rs_radius = max(self.cc.min_turn_radius, 0.5)

        # Reference path for reverse-curriculum start sampling.
        self._ref_path = _compute_reference_path(
            self.lot, self.cc, self.grid, self._goal_pose
        )
        if self._ref_path:
            cum = [0.0]
            for i in range(1, len(self._ref_path)):
                a, b = self._ref_path[i - 1], self._ref_path[i]
                cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
            self._ref_cumlen = cum
            self._build_ref_projection()
        else:
            self._ref_cumlen = None
            self._ref_proj_xy = None

    def _build_ref_projection(self, step: float = 0.2) -> None:
        """Densify the reference path into numpy arrays for fast projection
        (used by the path-based reward potential)."""
        pts: List[Tuple[float, float, float]] = [self._ref_path[0]]
        for a, b in zip(self._ref_path, self._ref_path[1:]):
            dist = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(math.ceil(dist / step)))
            for k in range(1, n + 1):
                t = k / n
                pts.append((
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + angle_diff(b[2], a[2]) * t,
                ))
        xy = np.array([(p[0], p[1]) for p in pts], dtype=np.float64)
        th = np.array([p[2] for p in pts], dtype=np.float64)
        seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self._ref_proj_xy = xy
        self._ref_proj_th = th
        self._ref_proj_remaining = cum[-1] - cum

    @property
    def curriculum_stage(self) -> int:
        return self._curriculum_stage_idx

    @curriculum_stage.setter
    def curriculum_stage(self, value: int) -> None:
        value = max(0, min(value, len(CURRICULUM_STAGES) - 1))
        if value != self._curriculum_stage_idx:
            self._curriculum_stage_idx = value
            self._rebuild_world()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_dim(self) -> int:
        # goal_x_ego, goal_y_ego, sin(dtheta), cos(dtheta), velocity, steering,
        # boundary_dist — all normalised to roughly [-1, 1]
        base = 7
        # K obstacles * 4 features each (dx_ego, dy_ego, w, h)
        obstacle_features = self.MAX_OBS_K * 4
        return base + obstacle_features

    def _get_obs_vector(self) -> np.ndarray:
        gx, gy, gtheta = self._goal_pose
        cx, cy, ctheta = self._car.x, self._car.y, self._car.theta

        # Goal position in the car's body frame (ego-centric): the policy
        # sees "where the goal is relative to me" independent of world frame.
        dx = gx - cx
        dy = gy - cy
        cos_t, sin_t = math.cos(ctheta), math.sin(ctheta)
        goal_x_ego = cos_t * dx + sin_t * dy
        goal_y_ego = -sin_t * dx + cos_t * dy
        dtheta = angle_diff(gtheta, ctheta)

        boundary_dist = self._min_boundary_distance((cx, cy, ctheta))

        obs = [
            goal_x_ego / 10.0,
            goal_y_ego / 10.0,
            math.sin(dtheta),
            math.cos(dtheta),
            self._velocity / self.max_speed,
            self._steering / self.cc.max_steer,
            min(boundary_dist, 3.0) / 3.0,
        ]

        # Nearest-K obstacle features relative to ego
        obs_features = self._obstacle_features(cos_t, sin_t)
        obs.extend(obs_features)

        return np.array(obs, dtype=np.float32)

    def _obstacle_features(self, cos_t: float, sin_t: float) -> List[float]:
        cx, cy = self._car.x, self._car.y
        features: List[Tuple[float, float, float, float, float]] = []

        for obs_rect in self._obstacles:
            ox, oy = obs_rect.center
            dx = ox - cx
            dy = oy - cy
            dist = math.hypot(dx, dy)
            ex = cos_t * dx + sin_t * dy
            ey = -sin_t * dx + cos_t * dy
            features.append((dist, ex, ey, obs_rect.w, obs_rect.h))

        features.sort(key=lambda t: t[0])
        result: List[float] = []
        for i in range(self.MAX_OBS_K):
            if i < len(features):
                _, ex, ey, w, h = features[i]
                result.extend([ex / 10.0, ey / 10.0, w / 5.0, h / 5.0])
            else:
                result.extend([0.0, 0.0, 0.0, 0.0])
        return result

    def _achieved_goal(self) -> np.ndarray:
        return np.array([
            self._car.x,
            self._car.y,
            math.cos(self._car.theta),
            math.sin(self._car.theta),
        ], dtype=np.float32)

    def _desired_goal(self) -> np.ndarray:
        gx, gy, gtheta = self._goal_pose
        return np.array([
            gx, gy, math.cos(gtheta), math.sin(gtheta)
        ], dtype=np.float32)

    def _make_obs(self) -> Any:
        if self.goal_conditioned:
            return {
                "observation": self._get_obs_vector(),
                "achieved_goal": self._achieved_goal(),
                "desired_goal": self._desired_goal(),
            }
        return self._get_obs_vector()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Any, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "start_pose" in options:
            start_pose = tuple(options["start_pose"])
        else:
            stage = CURRICULUM_STAGES[self._curriculum_stage_idx]
            start_pose = self._sample_start(stage)

        self._car = CarDynamics(start_pose[0], start_pose[1], start_pose[2], self.cc)
        self._velocity = 0.0
        self._steering = 0.0
        self._prev_steering = 0.0
        self._step_count = 0
        self._gear_switches = 0
        self._prev_velocity_sign = 0
        self._gear_switched_now = False

        gx, gy, gtheta = self._goal_pose
        self._prev_pos_error = math.hypot(
            self._car.x - gx, self._car.y - gy
        )
        self._prev_heading_error = abs(angle_diff(self._car.theta, gtheta))
        self._prev_dist_to_go = self._distance_to_go(start_pose)

        return self._make_obs(), self._get_info()

    def _rs_distance(self, pose: Pose) -> float:
        """Non-holonomic distance-to-go: Reeds-Shepp shortest path length.

        Unlike the Euclidean distance this *decreases* while driving forward
        past the spot to set up the reverse-in arc, so a potential based on
        it rewards the real maneuver instead of punishing it.
        """
        d = reeds_shepp.path_length(pose, self._goal_pose, self._rs_radius)
        if not math.isfinite(d):
            gx, gy, _ = self._goal_pose
            d = math.hypot(pose[0] - gx, pose[1] - gy)
        return d

    def _distance_to_go(self, pose: Pose) -> float:
        """Distance-to-go used as the reward potential.

        Projects the pose onto the (densified) hybrid A* reference path —
        nearest point by position + heading — and returns the remaining arc
        length plus deviation terms. Unlike the raw Reeds-Shepp distance this
        is boundary-aware (the reference avoids the walls) and has no plateau
        along the forward leg: RS distance is nearly flat between the spot
        and the gear-switch point because it ignores the lane boundary,
        which left the policy with no gradient exactly where it must commit
        to the overshoot.
        """
        if self._ref_proj_xy is None:
            return self._rs_distance(pose)
        dx = self._ref_proj_xy[:, 0] - pose[0]
        dy = self._ref_proj_xy[:, 1] - pose[1]
        d2 = dx * dx + dy * dy
        dth = self._ref_proj_th - pose[2]
        ang = np.abs(np.arctan2(np.sin(dth), np.cos(dth)))
        score = d2 + 2.0 * ang * ang
        i = int(np.argmin(score))
        # Deviation weights must exceed the remaining-arclen gained by
        # cutting inside a curve, or the potential rewards corner-cutting
        # (policies arrived at the spot over-rotated and wedged).
        d_path = float(
            self._ref_proj_remaining[i]
            + 2.5 * math.sqrt(d2[i])
            + 1.0 * ang[i]
        )
        # Near the goal, add the Reeds-Shepp maneuver length: it encodes the
        # exact "distance to perfectly aligned" (a 10 cm lateral error means
        # metres of RS correction), giving a much sharper end-game alignment
        # gradient than path deviation alone. Capped, so in the far field it
        # is a constant and the path term alone steers.
        if d_path < 9.0:
            rs_term = min(self._rs_distance(pose), 5.0)
        else:
            rs_term = 5.0
        return d_path + 0.8 * rs_term

    def _pose_on_ref_path(self, t: float) -> Pose:
        """Pose at arc-length fraction ``t`` (0=start, 1=goal) of the ref path."""
        cum = self._ref_cumlen
        target = t * cum[-1]
        # Linear scan is fine: paths are a few hundred points.
        for i in range(1, len(cum)):
            if cum[i] >= target:
                return self._ref_path[i]
        return self._ref_path[-1]

    def _sample_start(self, stage: CurriculumStage) -> Pose:
        """Sample a start pose for the stage's mixture:

        - fixed_start_ratio  → lot's fixed start pose + small noise
        - lane_random_ratio  → random in-lane approach pose
        - remainder          → reverse curriculum: a pose on the hybrid A*
                               reference path at fraction t ∈ [path_t_lo, 0.95]
        """
        r = self._rng.random()

        if r < stage.fixed_start_ratio or self._ref_path is None:
            fx, fy, ftheta = self.lot.car_start_pose
            for _ in range(20):
                pose = (
                    fx + self._rng.uniform(-0.8, 0.8),
                    fy + self._rng.uniform(-0.4, 0.4),
                    wrap_pi(ftheta + self._rng.uniform(
                        -math.radians(12), math.radians(12))),
                )
                if self.grid.pose_is_valid(pose):
                    return pose
            return self.lot.car_start_pose
        r -= stage.fixed_start_ratio

        if r < stage.lane_random_ratio:
            lane = self.lot.lane_rect
            for _ in range(50):
                pose = (
                    self._rng.uniform(lane.x + 0.5, lane.right - self.cc.length - 0.5),
                    self._rng.uniform(lane.y + 1.2, lane.top - 1.2),
                    wrap_pi(self._rng.uniform(-math.radians(30), math.radians(30))),
                )
                if self.grid.pose_is_valid(pose):
                    return pose
            return self.lot.car_start_pose

        # Reverse curriculum along the reference path. Cap t below 1 and
        # reject starts closer than 0.8 m to the goal: poses hard against
        # the spot's back wall are unwinnable (any reverse step collides)
        # and only pollute training and eval.
        gx, gy, _ = self._goal_pose
        for _ in range(20):
            t = self._rng.uniform(stage.path_t_lo, 0.90)
            px, py, ptheta = self._pose_on_ref_path(t)
            pose = (
                px + self._rng.uniform(-0.3, 0.3),
                py + self._rng.uniform(-0.3, 0.3),
                wrap_pi(ptheta + self._rng.uniform(
                    -math.radians(8), math.radians(8))),
            )
            if (math.hypot(pose[0] - gx, pose[1] - gy) >= 0.8
                    and self.grid.pose_is_valid(pose)):
                return pose
        return self._pose_on_ref_path(0.75)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        self._step_count += 1

        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        speed_cmd = float(np.clip(action[1], -1.0, 1.0))

        self._prev_steering = self._steering
        self._steering = steer_cmd * self.cc.max_steer
        self._velocity = speed_cmd * self.max_speed

        # Track gear switches
        self._gear_switched_now = False
        vel_sign = 1 if self._velocity > 0.01 else (-1 if self._velocity < -0.01 else 0)
        if (
            vel_sign != 0
            and self._prev_velocity_sign != 0
            and vel_sign != self._prev_velocity_sign
        ):
            self._gear_switches += 1
            self._gear_switched_now = True
        if vel_sign != 0:
            self._prev_velocity_sign = vel_sign

        # Advance dynamics
        self._car.step(self._velocity, self._steering, self.dt)
        self._car.theta = wrap_pi(self._car.theta)

        # Check termination conditions
        pose = (self._car.x, self._car.y, self._car.theta)
        collision = not self.grid.pose_is_valid(pose)
        success = self._check_success(pose)
        truncated = self._step_count >= self.max_episode_steps

        terminated = collision or success

        # Compute reward
        reward = self._compute_reward(pose, collision, success)

        return self._make_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _check_success(self, pose: Pose) -> bool:
        gx, gy, gtheta = self._goal_pose
        pos_error = math.hypot(pose[0] - gx, pose[1] - gy)
        heading_error = abs(angle_diff(pose[2], gtheta))
        if pos_error > 0.35 or heading_error > math.radians(10):
            return False
        return self.grid.pose_is_fully_in_spot(pose, margin=0.0)

    def _compute_reward(self, pose: Pose, collision: bool, success: bool) -> float:
        if self.reward_type == "sparse":
            return self._sparse_reward(collision, success)
        return self._dense_reward(pose, collision, success)

    def _sparse_reward(self, collision: bool, success: bool) -> float:
        if success:
            return 0.0
        if collision:
            return -1.0
        return -1.0

    def _min_boundary_distance(self, pose: Pose) -> float:
        """Minimum distance from any car corner to the nearest boundary WALL.

        Returns 0 if the car is outside the drivable area (collision).
        Boundaries = edges of (lane_rect union spot_rect), except the shared
        lane/spot opening, which is open space the car must drive through:
        there the relevant walls are the two opening corner posts, not the
        shared edge. (Treating the opening as a wall put a proximity-penalty
        barrier exactly across the spot threshold — policies learned to back
        in and stall at the opening rather than cross it.)
        """
        lane = self.lot.lane_rect
        spot = self.lot.spot_rect
        corners = self.lot.car_corners(pose)
        min_dist = float("inf")

        # Spot sits above the lane (perpendicular) or below it (parallel).
        spot_above = spot.y >= lane.top - 1e-9
        open_lo, open_hi = spot.x, spot.right
        open_y = lane.top if spot_above else lane.y
        posts = ((open_lo, open_y), (open_hi, open_y))

        def post_dist(cx: float, cy: float) -> float:
            return min(math.hypot(cx - px, cy - py) for px, py in posts)

        for cx, cy in corners:
            in_lane = (lane.x <= cx <= lane.right and lane.y <= cy <= lane.top)
            in_spot = (spot.x <= cx <= spot.right and spot.y <= cy <= spot.top)
            if not (in_lane or in_spot):
                return 0.0

            cands = []
            if in_lane:
                cands += [cx - lane.x, lane.right - cx]
                # Lane edge away from the spot is always a wall.
                cands.append(lane.top - cy if not spot_above else cy - lane.y)
                # Edge facing the spot: open within the opening x-range.
                facing = (lane.top - cy) if spot_above else (cy - lane.y)
                if open_lo <= cx <= open_hi:
                    cands.append(post_dist(cx, cy))
                else:
                    cands.append(facing)
            if in_spot:
                cands += [cx - spot.x, spot.right - cx]
                # Spot edge away from the lane (back wall) is always a wall.
                cands.append(spot.top - cy if spot_above else cy - spot.y)
                # Edge facing the lane: the opening.
                facing = (cy - spot.y) if spot_above else (spot.top - cy)
                if open_lo <= cx <= open_hi:
                    cands.append(post_dist(cx, cy))
                else:
                    cands.append(facing)
            min_dist = min(min_dist, min(cands))

        for obs_rect in self._obstacles:
            for cx, cy in corners:
                ox, oy = obs_rect.center
                hw, hh = obs_rect.w / 2, obs_rect.h / 2
                dx = max(0, abs(cx - ox) - hw)
                dy = max(0, abs(cy - oy) - hh)
                min_dist = min(min_dist, math.hypot(dx, dy))

        return min_dist

    def _dense_reward(self, pose: Pose, collision: bool, success: bool) -> float:
        gx, gy, gtheta = self._goal_pose

        pos_error = math.hypot(pose[0] - gx, pose[1] - gy)
        heading_error = abs(angle_diff(pose[2], gtheta))

        reward = 0.0

        # Potential-based shaping on the reference-path distance-to-go. This
        # is the single progress signal: it follows the full non-holonomic
        # maneuver (forward overshoot past the spot, reverse-in arc) that a
        # Euclidean potential actively punishes, and unlike raw Reeds-Shepp
        # it has no plateau along the forward leg.
        dist_to_go = self._distance_to_go(pose)
        reward += 2.0 * (self._prev_dist_to_go - dist_to_go)
        self._prev_dist_to_go = dist_to_go

        # Penalties: small, so they never dominate the progress signal
        reward += -0.02  # time (also discourages stalling mid-maneuver)
        reward += -0.01 * abs(self._steering / self.cc.max_steer)
        reward += -0.05 * abs(self._steering - self._prev_steering) / self.cc.max_steer
        if self._gear_switched_now:
            reward += -0.1

        # Proximity penalty: graded signal before collision. The margin is
        # below the in-spot lateral clearance (0.35 m for default sizes) so
        # being properly parked costs nothing.
        if not collision:
            boundary_dist = self._min_boundary_distance(pose)
            if boundary_dist < self.proximity_margin:
                fraction = 1.0 - (boundary_dist / self.proximity_margin)
                speed_factor = abs(self._velocity) / self.max_speed
                reward -= self.proximity_penalty_scale * fraction * (0.3 + 0.7 * speed_factor)

        # Terminal rewards
        if success:
            speed_at_goal = abs(self._velocity) / self.max_speed
            reward += 100.0 - 10.0 * speed_at_goal
        if collision:
            speed_factor = abs(self._velocity) / self.max_speed
            reward -= 50.0 + 25.0 * speed_factor

        # Update previous errors for next step
        self._prev_pos_error = pos_error
        self._prev_heading_error = heading_error

        return reward

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict,
    ) -> np.ndarray:
        """HER-compatible reward computation (vectorized)."""
        dx = achieved_goal[..., 0] - desired_goal[..., 0]
        dy = achieved_goal[..., 1] - desired_goal[..., 1]
        pos_error = np.sqrt(dx**2 + dy**2)

        # cos/sin angle difference
        cos_a, sin_a = achieved_goal[..., 2], achieved_goal[..., 3]
        cos_d, sin_d = desired_goal[..., 2], desired_goal[..., 3]
        heading_error = np.abs(np.arctan2(
            sin_a * cos_d - cos_a * sin_d,
            cos_a * cos_d + sin_a * sin_d,
        ))

        success = (pos_error < 0.35) & (heading_error < np.radians(10))
        if self.reward_type == "sparse":
            return np.where(success, 0.0, -1.0).astype(np.float32)

        reward = -2.0 * pos_error - 0.5 * heading_error
        reward = np.where(success, reward + 100.0, reward)
        return reward.astype(np.float32)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self) -> Dict[str, Any]:
        if self._car is None:
            return {}
        gx, gy, gtheta = self._goal_pose
        pose = (self._car.x, self._car.y, self._car.theta)
        pos_error = math.hypot(pose[0] - gx, pose[1] - gy)
        heading_error = abs(angle_diff(pose[2], gtheta))
        return {
            "pos_error": pos_error,
            "heading_error_deg": math.degrees(heading_error),
            "step": self._step_count,
            "gear_switches": self._gear_switches,
            "is_success": self._check_success(pose),
            "collision": not self.grid.pose_is_valid(pose),
        }

    # ------------------------------------------------------------------
    # Rendering (minimal — use simulation.py for full visualization)
    # ------------------------------------------------------------------

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
