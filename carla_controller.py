"""
Pure Pursuit closed-loop controller adapted for CARLA's VehicleControl API.

The Hybrid A* planner produces a sequence of `Waypoint(x, y, theta)` in the
planner frame. This module:

  1. Splits the planned path into single-gear segments (same gear-detection
     logic as `tracker.py`),
  2. Runs Pure Pursuit on each segment to compute (steering, throttle, brake,
     reverse) per simulation tick,
  3. Emits a `carla.VehicleControl` object that the demo loop applies to the
     ego vehicle.

The math is identical to `tracker.py` (including the reverse-steering-sign
flip), but the output is shaped for CARLA's actuator interface rather than a
synthetic bicycle integrator. The bicycle-model simulation that lived inside
`tracker.py` is now replaced by reading the vehicle's actual pose from CARLA
between ticks.

This module's classes do NOT import `carla` at module load time. The control
output is described by a small `ControlCommand` dataclass; the demo wraps it
into a real `carla.VehicleControl` only when CARLA is available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from config import CarConfig
from geom import (
    PURE_PURSUIT_L2_FLOOR as _L2_FLOOR,
    angle_diff as _angle_diff,
    closest_index as _closest_index,
    split_by_gear as _split_by_gear,
)
from trajectory import Waypoint


Pose = Tuple[float, float, float]


@dataclass
class ControlConfig:
    target_forward_speed: float = 0.9   # m/s — slow for parking maneuvers
    target_reverse_speed: float = 0.6
    lookahead_base: float = 0.55
    lookahead_gain: float = 0.20
    throttle_kp: float = 1.2            # throttle per m/s of error
    brake_kp: float = 1.5
    max_throttle: float = 0.45
    max_brake: float = 1.0
    cusp_brake_duration_s: float = 1.2  # hold long enough to actually stop
    segment_xy_tol: float = 0.35
    segment_theta_tol_deg: float = 14.0


@dataclass
class ControlCommand:
    """Actuator command shaped for CARLA's VehicleControl."""
    steer: float    # in [-1, +1] — fraction of max steering
    throttle: float # in [0, 1]
    brake: float    # in [0, 1]
    reverse: bool
    # diagnostics:
    target_idx: int = 0
    segment_idx: int = 0
    done: bool = False


class CarlaPurePursuitController:
    """
    Stateful Pure Pursuit controller that consumes the actual vehicle pose
    each tick and emits a ControlCommand.

    The controller advances through gear segments automatically. At each
    cusp it stops the vehicle for `cusp_brake_duration_s`, then engages the
    new gear. When the final segment terminates close to the goal pose the
    controller emits `done=True` commands (brake fully) forever.
    """

    def __init__(
        self,
        planned: Sequence[Waypoint],
        cc: CarConfig,
        cfg: Optional[ControlConfig] = None,
    ):
        self.cfg = cfg or ControlConfig()
        self.cc = cc
        self.segments = _split_by_gear(planned)
        if not self.segments:
            raise ValueError("CarlaPurePursuitController: planned path is empty")
        self.segment_idx = 0
        self._segment_target_idx = 0
        self._cusp_brake_remaining_s = 0.0
        self._done = False
        self._goal_pose = (planned[-1].x, planned[-1].y, planned[-1].theta)

    @property
    def done(self) -> bool:
        return self._done

    def _select_target(
        self,
        seg: Sequence[Waypoint],
        x: float,
        y: float,
        gear: int,
    ) -> Tuple[int, Waypoint]:
        # advance closest index in a small forward window
        best_i = _closest_index(seg, x, y, self._segment_target_idx, window=40)
        self._segment_target_idx = best_i

        Ld = self.cfg.lookahead_base + self.cfg.lookahead_gain * (
            self.cfg.target_forward_speed if gear > 0 else self.cfg.target_reverse_speed
        )
        target_i = best_i
        accum = 0.0
        last_idx = len(seg) - 1
        while target_i < last_idx and accum < Ld:
            a = seg[target_i]
            b = seg[target_i + 1]
            accum += math.hypot(b.x - a.x, b.y - a.y)
            target_i += 1
        return target_i, seg[target_i]

    def step(
        self,
        pose: Pose,
        current_speed_mps: float,
        dt: float,
    ) -> ControlCommand:
        """
        Pose = actual (x, y, theta) of the vehicle in planner frame, read
        from CARLA. current_speed_mps is the scalar speed magnitude.
        """
        if self._done:
            return ControlCommand(steer=0.0, throttle=0.0, brake=self.cfg.max_brake,
                                  reverse=False, done=True,
                                  segment_idx=self.segment_idx,
                                  target_idx=self._segment_target_idx)

        # Cusp brake: hold the vehicle still while transitioning gears.
        if self._cusp_brake_remaining_s > 0:
            self._cusp_brake_remaining_s -= dt
            return ControlCommand(steer=0.0, throttle=0.0, brake=self.cfg.max_brake,
                                  reverse=self.segments[self.segment_idx][0] < 0,
                                  segment_idx=self.segment_idx,
                                  target_idx=self._segment_target_idx)

        gear, seg = self.segments[self.segment_idx]
        x, y, theta = pose
        target_i, target = self._select_target(seg, x, y, gear)

        # Check if segment is finished.
        last_idx = len(seg) - 1
        seg_end = seg[-1]
        end_dist = math.hypot(x - seg_end.x, y - seg_end.y)
        end_heading = abs(_angle_diff(theta, seg_end.theta))

        # Detect overshoot: if the vehicle has passed seg_end along the
        # segment direction, treat the segment as done even if the heading
        # is slightly off. Without this we keep tracking forward forever
        # whenever we glide past the endpoint by more than segment_xy_tol.
        #
        # The reference direction is taken from the last waypoint at least
        # 0.5 m back along the segment (or seg_end's planned heading if the
        # segment is shorter than that). Using seg[-2] alone gave a noisy
        # projection when the smoother + densifier left a sub-decimetre
        # terminal sample.
        overshot = False
        if self._segment_target_idx >= last_idx:
            seg_dx, seg_dy = 0.0, 0.0
            for k in range(len(seg) - 2, -1, -1):
                cand_dx = seg_end.x - seg[k].x
                cand_dy = seg_end.y - seg[k].y
                if math.hypot(cand_dx, cand_dy) >= 0.5:
                    seg_dx, seg_dy = cand_dx, cand_dy
                    break
            seg_len = math.hypot(seg_dx, seg_dy)
            if seg_len < 1e-6:
                # Segment shorter than 0.5 m — fall back to the planned end
                # heading direction so the projection is well defined.
                seg_dx = math.cos(seg_end.theta)
                seg_dy = math.sin(seg_end.theta)
                if gear < 0:
                    seg_dx, seg_dy = -seg_dx, -seg_dy
                seg_len = 1.0
            advance = ((x - seg_end.x) * seg_dx + (y - seg_end.y) * seg_dy) / seg_len
            if advance > 0:
                overshot = True

        if self._segment_target_idx >= last_idx and (
            (end_dist < self.cfg.segment_xy_tol
             and end_heading < math.radians(self.cfg.segment_theta_tol_deg))
            or overshot
        ):
            self.segment_idx += 1
            self._segment_target_idx = 0
            if self.segment_idx >= len(self.segments):
                self._done = True
                return ControlCommand(
                    steer=0.0, throttle=0.0, brake=self.cfg.max_brake,
                    reverse=False, done=True,
                    segment_idx=self.segment_idx - 1,
                    target_idx=last_idx,
                )
            # Brief brake at cusp before engaging next gear
            self._cusp_brake_remaining_s = self.cfg.cusp_brake_duration_s
            return ControlCommand(
                steer=0.0, throttle=0.0, brake=self.cfg.max_brake,
                reverse=self.segments[self.segment_idx][0] < 0,
                segment_idx=self.segment_idx,
                target_idx=0,
            )

        # Pure pursuit steering geometry.
        dx = target.x - x
        dy = target.y - y
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        lx = cos_t * dx + sin_t * dy
        ly = -sin_t * dx + cos_t * dy
        if gear < 0:
            lx = -lx
            ly = -ly

        L2 = lx * lx + ly * ly
        # Floor L2 so 2*ly/L2 stays bounded when the lookahead vector is
        # essentially under the rear axle (numerical noise at segment
        # terminus). Cap implied |curvature| at 2/min_ld via the floor; at
        # min_ld = 0.05 m this allows |curvature| ≤ 40 m^-1 — well above any
        # real maneuver but avoids single-tick steering snaps.
        L2 = max(L2, _L2_FLOOR)
        curvature = 2.0 * ly / L2
        delta_rad = math.atan(self.cc.wheelbase * curvature)
        if gear < 0:
            delta_rad = -delta_rad

        delta_rad = max(-self.cc.max_steer, min(self.cc.max_steer, delta_rad))
        steer_norm = delta_rad / self.cc.max_steer  # [-1, 1]

        # Longitudinal: simple proportional throttle / brake to target speed.
        target_speed = (
            self.cfg.target_forward_speed if gear > 0 else self.cfg.target_reverse_speed
        )
        speed_error = target_speed - current_speed_mps
        throttle = 0.0
        brake = 0.0
        if speed_error > 0:
            throttle = min(self.cfg.max_throttle, self.cfg.throttle_kp * speed_error)
        else:
            brake = min(self.cfg.max_brake, self.cfg.brake_kp * (-speed_error))

        return ControlCommand(
            steer=steer_norm,
            throttle=throttle,
            brake=brake,
            reverse=gear < 0,
            target_idx=target_i,
            segment_idx=self.segment_idx,
        )


def control_to_carla(cmd: ControlCommand):
    """
    Wrap a ControlCommand into a `carla.VehicleControl`.

    Imported lazily so this module is usable without CARLA.
    """
    import carla
    return carla.VehicleControl(
        throttle=float(max(0.0, min(1.0, cmd.throttle))),
        steer=float(max(-1.0, min(1.0, cmd.steer))),
        brake=float(max(0.0, min(1.0, cmd.brake))),
        reverse=bool(cmd.reverse),
    )
