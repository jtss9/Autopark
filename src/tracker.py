"""
Pure Pursuit closed-loop tracker for kinematic bicycle vehicle.

The Hybrid A* planner emits paths with discrete forward/reverse gear changes.
We split the planned path into single-gear segments and run a Pure Pursuit
controller on each segment independently, coming to a halt at every cusp
(direction-reversal point). Each per-segment run terminates when the car
reaches the end of that segment, and the next segment then starts from the
actual achieved pose. The accumulated executed trajectory is reported back
together with cross-track tracking-error statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from config import CarConfig
from geom import angle_diff as _angle_diff, split_by_gear as _split_by_gear
from parking_lot import ParkingLot
from trajectory import Waypoint


@dataclass
class TrackerConfig:
    dt: float = 0.05
    forward_speed: float = 1.2
    reverse_speed: float = 0.8
    lookahead_base: float = 0.45
    lookahead_gain: float = 0.18
    segment_xy_tol: float = 0.12
    segment_theta_tol: float = math.radians(8.0)
    max_steps_per_segment: int = 2500


@dataclass
class TrackerResult:
    executed: List[Waypoint]
    succeeded: bool
    message: str
    mean_cte_m: float
    max_cte_m: float
    final_pos_error_m: float
    final_heading_error_deg: float
    fully_in_spot: bool
    cusps: int


def _cross_track_error(
    seg: Sequence[Waypoint],
    x: float,
    y: float,
    i: int,
) -> float:
    if i + 1 >= len(seg):
        wp = seg[i]
        return math.hypot(x - wp.x, y - wp.y)
    a = seg[i]
    b = seg[i + 1]
    seg_len = math.hypot(b.x - a.x, b.y - a.y)
    if seg_len < 1e-6:
        return math.hypot(x - a.x, y - a.y)
    return abs((b.x - a.x) * (a.y - y) - (a.x - x) * (b.y - a.y)) / seg_len


def _closest_index(seg: Sequence[Waypoint], x: float, y: float, start: int) -> int:
    end = min(len(seg), start + 40)
    best_i, best_d = start, float("inf")
    for i in range(start, end):
        wp = seg[i]
        d = (wp.x - x) ** 2 + (wp.y - y) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _track_segment(
    seg: Sequence[Waypoint],
    gear: int,
    pose: Tuple[float, float, float],
    cc: CarConfig,
    cfg: TrackerConfig,
) -> Tuple[List[Waypoint], List[float], Tuple[float, float, float], bool]:
    x, y, theta = pose
    executed: List[Waypoint] = []
    cte_samples: List[float] = []

    speed = cfg.forward_speed if gear > 0 else cfg.reverse_speed
    Ld = cfg.lookahead_base + cfg.lookahead_gain * speed
    target_seg_end = seg[-1]

    idx = 0
    last_idx = len(seg) - 1
    for _ in range(cfg.max_steps_per_segment):
        idx = _closest_index(seg, x, y, idx)

        # Pick a lookahead target ahead along the segment by arc length Ld.
        target_i = idx
        accum = 0.0
        while target_i < last_idx and accum < Ld:
            a = seg[target_i]
            b = seg[target_i + 1]
            accum += math.hypot(b.x - a.x, b.y - a.y)
            target_i += 1
        target = seg[target_i]

        dx = target.x - x
        dy = target.y - y
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        lx = cos_t * dx + sin_t * dy
        ly = -sin_t * dx + cos_t * dy
        # For reverse pursuit, treat the rear of the car as the new "front":
        # negate the lookahead vector AND negate the resulting steering, since
        # in the bicycle model theta_dot = v * tan(delta) / L flips sign with v.
        if gear < 0:
            lx = -lx
            ly = -ly

        L2 = lx * lx + ly * ly
        # Floor L2 to a small minimum so 2*ly/L2 stays bounded when the
        # target lookahead is essentially under the rear axle. Without the
        # floor a 1e-3 m × 1e-4 m lookahead produces curvature ~ 90 m^-1 and
        # saturates the wheel for one tick. The floor caps the implied
        # |curvature| at 2/min_ld; at min_ld = 0.05 m that allows up to
        # 40 m^-1, well above any physical maneuver but smooth.
        L2 = max(L2, 0.0025)  # 0.05 m floor
        curvature = 2.0 * ly / L2
        delta = math.atan(cc.wheelbase * curvature)
        if gear < 0:
            delta = -delta
        delta = max(-cc.max_steer, min(cc.max_steer, delta))

        v = gear * speed
        x += v * math.cos(theta) * cfg.dt
        y += v * math.sin(theta) * cfg.dt
        theta = _angle_diff(theta + v * math.tan(delta) / cc.wheelbase * cfg.dt, 0.0)

        executed.append(Waypoint(x, y, theta))
        cte_samples.append(_cross_track_error(seg, x, y, idx))

        # Stop the segment once we are close to its terminal waypoint AND on
        # the last index, or once we have clearly passed it.
        end_dist = math.hypot(x - target_seg_end.x, y - target_seg_end.y)
        end_heading = abs(_angle_diff(theta, target_seg_end.theta))
        if idx >= last_idx and end_dist < cfg.segment_xy_tol:
            return executed, cte_samples, (x, y, theta), True
        if idx >= last_idx and end_dist > cfg.segment_xy_tol * 4:
            # Overshot far past the endpoint. Only accept the segment if the
            # heading is still close to the planned terminal heading; otherwise
            # report the segment as failed so all_segments_ok flips to False
            # and the tracker message reflects the real divergence.
            # cfg.segment_theta_tol is already in radians; allow 2x tolerance
            # for the overshoot bail since the geometry is necessarily looser.
            seg_ok = end_heading < cfg.segment_theta_tol * 2
            return executed, cte_samples, (x, y, theta), seg_ok

    return executed, cte_samples, (x, y, theta), False


def track_path(
    planned: Sequence[Waypoint],
    cc: CarConfig,
    lot: ParkingLot,
    tcfg: Optional[TrackerConfig] = None,
) -> TrackerResult:
    if len(planned) < 2:
        return TrackerResult(
            list(planned), False, "Tracker: planned path is too short.",
            0.0, 0.0, 0.0, 0.0, False, 0,
        )

    cfg = tcfg or TrackerConfig()
    segments = _split_by_gear(planned)
    if not segments:
        return TrackerResult(
            list(planned), False, "Tracker: no drivable segments.",
            0.0, 0.0, 0.0, 0.0, False, 0,
        )

    executed: List[Waypoint] = [Waypoint(planned[0].x, planned[0].y, planned[0].theta)]
    cte_all: List[float] = []
    pose = (planned[0].x, planned[0].y, planned[0].theta)
    all_segments_ok = True

    for gear, seg in segments:
        seg_exec, seg_cte, pose, seg_ok = _track_segment(seg, gear, pose, cc, cfg)
        executed.extend(seg_exec)
        cte_all.extend(seg_cte)
        if not seg_ok:
            all_segments_ok = False
            break

    final = executed[-1]
    goal = planned[-1]
    pos_err = math.hypot(final.x - goal.x, final.y - goal.y)
    heading_err = abs(_angle_diff(final.theta, goal.theta))

    spot = lot.spot_rect
    in_spot = all(
        spot.x <= cx <= spot.right and spot.y <= cy <= spot.top
        for cx, cy in lot.car_corners((final.x, final.y, final.theta))
    )

    mean_cte = sum(cte_all) / len(cte_all) if cte_all else 0.0
    max_cte = max(cte_all) if cte_all else 0.0

    message = "Tracker: OK" if all_segments_ok else "Tracker: a segment failed to converge"
    success = all_segments_ok and in_spot
    if all_segments_ok and not in_spot:
        message = "Tracker: executed pose is not fully inside the parking spot."

    return TrackerResult(
        executed=executed,
        succeeded=success,
        message=message,
        mean_cte_m=mean_cte,
        max_cte_m=max_cte,
        final_pos_error_m=pos_err,
        final_heading_error_deg=math.degrees(heading_err),
        fully_in_spot=in_spot,
        cusps=max(0, len(segments) - 1),
    )
