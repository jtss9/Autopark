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
from hybrid_astar import OccupancyGrid, perpendicular_goal_pose, parallel_goal_pose
from parking_lot import ParkingLot, Rect
from scenarios import obstacles_for


Pose = Tuple[float, float, float]


@dataclass
class CurriculumStage:
    name: str
    max_start_distance: float
    max_start_heading_error: float
    obstacle_scenario: str = "none"
    success_rate_threshold: float = 0.75


CURRICULUM_STAGES = [
    CurriculumStage("close", 4.0, math.radians(45), "none", 0.50),
    CurriculumStage("medium", 7.0, math.radians(90), "none", 0.40),
    CurriculumStage("far", 11.0, math.pi, "none", 0.30),
    CurriculumStage("simple_obs", 11.0, math.pi, "entry_blocker", 0.25),
    CurriculumStage("hard_obs", 11.0, math.pi, "parked_cars", 0.20),
]


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
        dt: float = 0.1,
        max_speed: float = 2.0,
        goal_conditioned: bool = False,
        reward_type: str = "dense",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        fixed_start_ratio: float = 0.6,
        proximity_margin: float = 0.8,
        proximity_penalty_scale: float = 2.0,
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
        # dx, dy, sin(dtheta), cos(dtheta), velocity, steering, boundary_dist
        base = 7
        # K obstacles * 4 features each (dx, dy, w, h)
        obstacle_features = self.MAX_OBS_K * 4
        return base + obstacle_features

    def _get_obs_vector(self) -> np.ndarray:
        gx, gy, gtheta = self._goal_pose
        cx, cy, ctheta = self._car.x, self._car.y, self._car.theta

        dx = cx - gx
        dy = cy - gy
        dtheta = angle_diff(ctheta, gtheta)

        boundary_dist = self._min_boundary_distance(
            (cx, cy, ctheta)
        )

        obs = [
            dx,
            dy,
            math.sin(dtheta),
            math.cos(dtheta),
            self._velocity,
            self._steering,
            boundary_dist,
        ]

        # Nearest-K obstacle features relative to ego
        obs_features = self._obstacle_features()
        obs.extend(obs_features)

        return np.array(obs, dtype=np.float32)

    def _obstacle_features(self) -> List[float]:
        cx, cy = self._car.x, self._car.y
        features: List[Tuple[float, float, float, float, float]] = []

        for obs_rect in self._obstacles:
            ox, oy = obs_rect.center
            dx = ox - cx
            dy = oy - cy
            dist = math.hypot(dx, dy)
            features.append((dist, dx, dy, obs_rect.w, obs_rect.h))

        features.sort(key=lambda t: t[0])
        result: List[float] = []
        for i in range(self.MAX_OBS_K):
            if i < len(features):
                _, dx, dy, w, h = features[i]
                result.extend([dx, dy, w, h])
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

        gx, gy, gtheta = self._goal_pose
        self._prev_pos_error = math.hypot(
            self._car.x - gx, self._car.y - gy
        )
        self._prev_heading_error = abs(angle_diff(self._car.theta, gtheta))

        spot_entry_x = gx
        spot_entry_y = self.lot.lane_rect.h / 2
        self._prev_entry_dist = math.hypot(
            self._car.x - spot_entry_x, self._car.y - spot_entry_y
        )

        return self._make_obs(), self._get_info()

    def _sample_start(self, stage: CurriculumStage) -> Pose:
        """Sample a valid start pose within curriculum constraints.

        With probability ``fixed_start_ratio``, samples near the lot's
        default car_start_pose with small perturbations (±1m position,
        ±15° heading) so the policy learns the realistic approach while
        staying robust to slight variations.
        """
        if self._rng.random() < self.fixed_start_ratio:
            fx, fy, ftheta = self.lot.car_start_pose
            x = fx + self._rng.uniform(-1.0, 1.0)
            y = fy + self._rng.uniform(-0.5, 0.5)
            theta = wrap_pi(ftheta + self._rng.uniform(
                -math.radians(15), math.radians(15)
            ))
            pose = (x, y, theta)
            if self.grid.pose_is_valid(pose):
                return pose
            return self.lot.car_start_pose

        gx, gy, gtheta = self._goal_pose

        for _ in range(100):
            dist = self._rng.uniform(1.5, stage.max_start_distance)
            angle = self._rng.uniform(-math.pi, math.pi)
            x = gx + dist * math.cos(angle)
            y = gy + dist * math.sin(angle)
            heading_offset = self._rng.uniform(
                -stage.max_start_heading_error,
                stage.max_start_heading_error,
            )
            theta = wrap_pi(gtheta + heading_offset)
            pose = (x, y, theta)
            if self.grid.pose_is_valid(pose):
                return pose

        return self.lot.car_start_pose

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
        vel_sign = 1 if self._velocity > 0.01 else (-1 if self._velocity < -0.01 else 0)
        if (
            vel_sign != 0
            and self._prev_velocity_sign != 0
            and vel_sign != self._prev_velocity_sign
        ):
            self._gear_switches += 1
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
        """Minimum distance from any car corner to the nearest boundary edge.

        Returns 0 if the car is outside the drivable area (collision).
        Boundaries = edges of (lane_rect union spot_rect).
        """
        lane = self.lot.lane_rect
        spot = self.lot.spot_rect
        corners = self.lot.car_corners(pose)
        min_dist = float("inf")

        for cx, cy in corners:
            in_lane = (lane.x <= cx <= lane.right and lane.y <= cy <= lane.top)
            in_spot = (spot.x <= cx <= spot.right and spot.y <= cy <= spot.top)
            if not (in_lane or in_spot):
                return 0.0

            if in_lane:
                d = min(cx - lane.x, lane.right - cx, cy - lane.y, lane.top - cy)
            else:
                d = min(cx - spot.x, spot.right - cx, cy - spot.y, spot.top - cy)
            min_dist = min(min_dist, d)

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

        # Progress reward (potential-based)
        pos_progress = self._prev_pos_error - pos_error
        heading_progress = self._prev_heading_error - heading_error
        reward += 5.0 * pos_progress
        reward += 1.0 * heading_progress

        # Intermediate waypoint: reward getting to the lane position in front
        # of the spot (x ≈ goal_x, y ≈ lane center). This bridges the 11m gap
        # by giving signal before the car reaches the final goal.
        spot_entry_x = gx
        spot_entry_y = self.lot.lane_rect.h / 2  # lane center
        dist_to_entry = math.hypot(pose[0] - spot_entry_x, pose[1] - spot_entry_y)
        if not hasattr(self, '_prev_entry_dist'):
            self._prev_entry_dist = dist_to_entry
        entry_progress = self._prev_entry_dist - dist_to_entry
        reward += 2.0 * entry_progress
        self._prev_entry_dist = dist_to_entry

        # Phase-aware rewards: detect which phase the car is in
        # Phase 1: approaching spot entrance (far from goal, in lane)
        # Phase 2: at spot entrance, aligning (near spot x, in lane)
        # Phase 3: reversing into spot (y > lane top)
        spot_entry_x = self._goal_pose[0]
        at_spot_x = abs(pose[0] - spot_entry_x) < 2.0
        in_lane = pose[1] < self.lot.lane_rect.top
        heading_toward_spot = abs(angle_diff(pose[2], self._goal_pose[2])) < math.radians(45)

        if at_spot_x and heading_toward_spot and not in_lane:
            # Phase 3: reversing into spot — strong reward for y-progress
            reward += 3.0 * max(0, pose[1] - self.lot.lane_rect.top) / self.lot.spot_rect.h

        # Near-goal heading bonus
        if pos_error < 1.0:
            reward += -2.0 * heading_error

        # Penalties
        reward += -0.02  # time penalty
        reward += -0.01 * abs(self._steering / self.cc.max_steer)
        reward += -0.05 * abs(self._steering - self._prev_steering) / self.cc.max_steer
        reward += -0.005 * abs(self._velocity / self.max_speed)

        # Proximity penalty: gradient signal before collision (literature: proximity-graded)
        if not collision:
            boundary_dist = self._min_boundary_distance(pose)
            if boundary_dist < self.proximity_margin:
                fraction = 1.0 - (boundary_dist / self.proximity_margin)
                speed_factor = abs(self._velocity) / self.max_speed
                reward -= self.proximity_penalty_scale * fraction * (0.5 + 0.5 * speed_factor)

        # Terminal rewards
        if success:
            speed_at_goal = abs(self._velocity) / self.max_speed
            reward += 100.0 - 10.0 * speed_at_goal
        if collision:
            speed_factor = abs(self._velocity) / self.max_speed
            reward -= 50.0 + 50.0 * speed_factor

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
