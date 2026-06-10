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
    # Domain randomization: sample scene sizes (lane/spot/car) from the
    # feasible pool, and optionally drop one predefined obstacle into the
    # lane (worlds are admitted only if hybrid A* still finds a path, so
    # parking is possible by construction).
    randomize_scene: bool = False
    random_obstacle: bool = False
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
    # Stage 3: random in-lane approach poses + randomized scene sizes.
    CurriculumStage("robust", path_t_lo=0.0, fixed_start_ratio=0.40,
                    lane_random_ratio=0.20, success_rate_threshold=0.50,
                    randomize_scene=True),
    # Stage 4: randomized scenes with one obstacle in the lane.
    CurriculumStage("obstacle", path_t_lo=0.0, fixed_start_ratio=0.40,
                    lane_random_ratio=0.10, success_rate_threshold=0.40,
                    randomize_scene=True, random_obstacle=True),
]


# ---------------------------------------------------------------------------
# Domain-randomization pools
# ---------------------------------------------------------------------------

# Matches the Settings-UI slider ranges (settings_window.SLIDER_DEFS) plus
# the config.py defaults, so the deployed model covers what users can set.
_SCENE_POOL_CACHE: Dict[str, List[Tuple[float, float, float, float, float]]] = {}

# Fixed-size obstacle box, identical to the draggable UI obstacle.
OBSTACLE_SIZE = 0.9


def scene_pool(parking_type: str) -> List[Tuple[float, float, float, float, float]]:
    """Deterministic pool of (lane_w, spot_len, spot_w, car_len, car_w).

    Anchored on the Settings-UI defaults and the config.py defaults, plus a
    seeded sample of the slider ranges, pre-filtered by simple clearance
    rules. Final feasibility (hybrid A* finds a path) is checked lazily the
    first time a combo is used.
    """
    if parking_type in _SCENE_POOL_CACHE:
        return _SCENE_POOL_CACHE[parking_type]

    lanes = [3.5, 4.0, 4.4, 5.0, 5.5, 6.0]
    spot_lens = [5.0, 5.5, 6.0]
    spot_ws = [2.2, 2.5, 2.8, 3.0]
    cars = [(3.5, 1.6), (4.0, 1.7), (4.2, 1.8), (4.5, 1.8), (4.8, 2.0), (5.0, 2.2)]
    anchors = [
        (4.4, 5.5, 2.5, 4.2, 1.8),   # Settings-UI slider defaults
        (6.0, 6.0, 2.5, 4.5, 1.8),   # config.py defaults
    ]

    rng = np.random.default_rng(12345)
    pool: List[Tuple[float, float, float, float, float]] = list(anchors)
    seen = set(anchors)
    attempts = 0
    while len(pool) < 26 and attempts < 500:
        attempts += 1
        combo = (
            float(rng.choice(lanes)), float(rng.choice(spot_lens)),
            float(rng.choice(spot_ws)),
            *map(float, cars[rng.integers(len(cars))]),
        )
        lane_w, spot_l, spot_w, car_l, car_w = combo
        if combo in seen:
            continue
        # Lateral clearance must leave room for the success window.
        if spot_w - car_w < 0.45:
            continue
        # Depth clearance (perpendicular: car length vs spot length;
        # parallel: car must fit lengthwise with entry slack).
        if parking_type == "perpendicular" and spot_l - car_l < 0.6:
            continue
        if parking_type == "parallel" and spot_l - car_l < 0.7:
            continue
        seen.add(combo)
        pool.append(combo)

    _SCENE_POOL_CACHE[parking_type] = pool
    return pool


def obstacle_candidates(
    parking_type: str, scene: Tuple[float, float, float, float, float],
) -> List[Tuple[float, float, float, float]]:
    """Predefined single-obstacle placements (x, y, w, h) for a scene.

    Positions are anchored to the lot geometry (approach, under/over the
    spot, past the spot). Infeasible placements are weeded out lazily by
    the reference-path check, which guarantees the remaining worlds are
    parkable without collision.
    """
    lane_w, spot_l, spot_w = scene[0], scene[1], scene[2]
    s = OBSTACLE_SIZE
    if parking_type == "perpendicular":
        lane_len = ParkingLot._PERP_LANE_LEN
        spot_x = (lane_len - spot_w) / 2
        spot_right = spot_x + spot_w
        centers = [
            (spot_x - 3.5, 0.28 * lane_w),
            (spot_x - 3.5, 0.72 * lane_w),
            (spot_x + spot_w / 2, 0.22 * lane_w),
            (spot_right + 2.5, 0.28 * lane_w),
            (spot_right + 2.5, 0.72 * lane_w),
            (spot_x - 1.0, 0.75 * lane_w),
        ]
    else:
        lane_len = ParkingLot._PAR_LANE_LEN
        spot_x = (lane_len - spot_l) / 2
        spot_right = spot_x + spot_l
        centers = [
            (spot_x - 3.0, 0.70 * lane_w),
            (spot_x + spot_l / 2, 0.75 * lane_w),
            (spot_right + 2.0, 0.30 * lane_w),
            (spot_right + 2.0, 0.70 * lane_w),
            (spot_x - 1.0, 0.72 * lane_w),
        ]
    out = []
    for cx, cy in centers:
        # Keep the box fully inside the lane band.
        cy = min(max(cy, s / 2 + 0.05), lane_w - s / 2 - 0.05)
        out.append((round(cx - s / 2, 2), round(cy - s / 2, 2), s, s))
    return out


# Reference paths are expensive (hybrid A*); cache per world configuration,
# persisted to disk so feasibility checks for randomized worlds (including
# failed searches, which are the slow case) are paid once ever, not once
# per process.
_REF_PATH_CACHE: Dict[Tuple, Optional[List[Pose]]] = {}
_REF_PATH_CACHE_FILE = (
    __import__("pathlib").Path(__file__).parent.parent
    / "checkpoints" / "refpath_cache.json"
)
_REF_PATH_CACHE_LOADED = False


def _load_ref_path_cache() -> None:
    global _REF_PATH_CACHE_LOADED
    if _REF_PATH_CACHE_LOADED:
        return
    _REF_PATH_CACHE_LOADED = True
    try:
        import json
        with open(_REF_PATH_CACHE_FILE) as f:
            raw = json.load(f)
        for k, path in raw.items():
            key = eval(k)  # keys are repr() of plain tuples written by us
            _REF_PATH_CACHE[key] = (
                [tuple(p) for p in path] if path is not None else None
            )
    except (OSError, ValueError, SyntaxError):
        pass


def _save_ref_path_cache() -> None:
    try:
        import json
        _REF_PATH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_REF_PATH_CACHE_FILE, "w") as f:
            json.dump({repr(k): v for k, v in _REF_PATH_CACHE.items()}, f)
    except OSError:
        pass


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
    _load_ref_path_cache()
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
    _save_ref_path_cache()
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
        # dt 0.1 / max_speed 2.0 is the regime every successful checkpoint
        # and demonstration was produced in. Do not change it for training
        # runs that warm-start from existing checkpoints: resuming training
        # at a different dt rescales per-step rewards and transition
        # strides, the critic's value scale is wrong everywhere, and the
        # policy is destroyed within ~15k steps regardless of lr.
        dt: float = 0.1,
        max_speed: float = 2.0,
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

        # Domain randomization state. self.pc / self._base_cc hold the
        # constructor (deployment) configuration; randomized stages swap
        # the active world per episode. Worlds proven infeasible (no
        # reference path) are blacklisted.
        self._base_cc = self.cc
        self._world_key: Any = ("<uninitialised>",)
        self._dead_worlds = set()

        # Build world geometry (base world)
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

    def _rebuild_world(
        self,
        scene: Optional[Tuple[float, float, float, float, float]] = None,
        obstacle: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """(Re)build the active world.

        ``scene=None`` restores the base (constructor) configuration with
        its own obstacle, if any; otherwise the scene tuple
        (lane_w, spot_len, spot_w, car_len, car_w) plus the optional single
        obstacle define the world. Returns True iff a reference path
        exists, i.e. parking is possible without collision.
        """
        key = (scene, obstacle)
        if key == self._world_key:
            return self._ref_path is not None
        self._world_key = key

        stage = CURRICULUM_STAGES[self._curriculum_stage_idx]
        if scene is None:
            lane_w, spot_l, spot_w = (
                self.pc.lane_width, self.pc.spot_length, self.pc.spot_width)
            self.cc = self._base_cc
            active_obstacle = self.pc.obstacle
        else:
            lane_w, spot_l, spot_w, car_l, car_w = scene
            self.cc = CarConfig(length=car_l, width=car_w)
            active_obstacle = obstacle

        pc = ParkingConfig(
            lane_width=lane_w,
            spot_length=spot_l,
            spot_width=spot_w,
            parking_type=self.pc.parking_type,
            obstacle_scenario=stage.obstacle_scenario,
            planner=self.pc.planner,
            obstacle=active_obstacle,
        )
        self._active_pc = pc
        self.lot = ParkingLot(pc, self.cc)
        obstacles = obstacles_for(self.lot)
        if pc.obstacle is not None:
            x, y, w, h = pc.obstacle
            obstacles.append(Rect(x, y, w, h))
        self._obstacles = obstacles
        self.grid = OccupancyGrid(self.lot, obstacles=obstacles)

        # Proximity margin must stay below the in-spot lateral clearance of
        # the active scene, or being correctly parked gets penalised.
        clearance = (pc.spot_width - self.cc.width) / 2
        self._active_proximity_margin = min(
            self.proximity_margin, max(0.05, clearance - 0.02))

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
        return self._ref_path is not None

    def set_world(
        self,
        scene: Optional[Tuple[float, float, float, float, float]] = None,
        obstacle: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """Public world switch (used by demo collection and evaluation).
        Returns False if the world is infeasible (no reference path)."""
        return self._rebuild_world(scene, obstacle)

    def _select_world(self, stage: CurriculumStage) -> None:
        """Pick this episode's world according to the stage settings.

        Randomized worlds are admitted only if the reference-path check
        passes, so every training episode is parkable by construction.
        """
        if not stage.randomize_scene:
            self._rebuild_world()
            return

        # Rehearsal: keep ~40% of episodes on the base world so adapting to
        # randomized scenes does not catastrophically forget the deployment
        # scenario (warm-started runs lost the base behaviour within 15k
        # steps without this + a buffer prefilled with base experience).
        if self._rng.random() < 0.40:
            self._rebuild_world()
            return

        pool = scene_pool(self.pc.parking_type)
        # Obstacle worlds draw from a smaller scene subset to bound the
        # number of distinct (scene, obstacle) hybrid A* plans.
        scene_choices = pool[:8] if stage.random_obstacle else pool
        for _ in range(15):
            scene = scene_choices[int(self._rng.integers(len(scene_choices)))]
            obstacle = None
            if stage.random_obstacle:
                cands = obstacle_candidates(self.pc.parking_type, scene)
                obstacle = cands[int(self._rng.integers(len(cands)))]
            key = (scene, obstacle)
            if key in self._dead_worlds:
                continue
            if self._rebuild_world(scene, obstacle):
                return
            self._dead_worlds.add(key)
        # Fallback: base world is always feasible.
        self._rebuild_world()

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
            # Explicit start pose (deployment rollouts, fixed-start evals):
            # keep the current world so callers can pair set_world() with a
            # chosen start; never randomize underneath them.
            start_pose = tuple(options["start_pose"])
        else:
            stage = CURRICULUM_STAGES[self._curriculum_stage_idx]
            self._select_world(stage)
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
        t_lo = stage.path_t_lo
        # Parallel parking's hard part is the final in-spot straightening
        # (a multi-point wiggle); focus stage 0 on exactly that skill.
        if self.pc.parking_type == "parallel" and self._curriculum_stage_idx == 0:
            t_lo = max(t_lo, 0.75)
        gx, gy, _ = self._goal_pose
        for _ in range(20):
            t = self._rng.uniform(t_lo, 0.90)
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
            margin = self._active_proximity_margin
            boundary_dist = self._min_boundary_distance(pose)
            if boundary_dist < margin:
                fraction = 1.0 - (boundary_dist / margin)
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
