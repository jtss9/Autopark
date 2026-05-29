"""
Shared geometry helpers used by the planner, controllers, evaluator, and
CARLA bridge.

Centralising these avoids the multi-file drift that previously had
`_angle_diff` redefined in 7 modules and `_split_by_gear` defined in 2.

All helpers operate on planner-frame poses: (x, y, theta) with +x right,
+y up, theta in radians.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b wrapped into [-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def wrap_pi(theta: float) -> float:
    """Wrap theta into [-pi, pi]. Equivalent to angle_diff(theta, 0)."""
    return (theta + math.pi) % (2 * math.pi) - math.pi


def split_by_gear(planned) -> List[Tuple[int, list]]:
    """Split a planned-waypoint sequence into single-gear sub-segments.

    Each cusp (direction reversal) starts a new segment. Gear is inferred
    from the sign of (planner heading) projected onto (next-waypoint - this)
    displacement. Identical to the previous duplicates in tracker.py and
    carla_controller.py.

    `planned` may be any sequence of objects with .x, .y, .theta attributes.
    Returns a list of (gear, segment_list) pairs.
    """
    segments: List[Tuple[int, list]] = []
    if len(planned) < 2:
        return segments
    current_gear: Optional[int] = None
    current_seg: list = [planned[0]]
    for a, b in zip(planned, planned[1:]):
        dx = b.x - a.x
        dy = b.y - a.y
        if math.hypot(dx, dy) < 1e-6:
            current_seg.append(b)
            continue
        heading_dot = math.cos(a.theta) * dx + math.sin(a.theta) * dy
        gear = 1 if heading_dot >= 0 else -1
        if current_gear is None:
            current_gear = gear
        if gear != current_gear:
            segments.append((current_gear, current_seg))
            current_gear = gear
            current_seg = [a]
        current_seg.append(b)
    if current_gear is not None and len(current_seg) >= 2:
        segments.append((current_gear, current_seg))
    return segments


def path_length(waypoints: Sequence) -> float:
    """Sum of Euclidean distances between consecutive waypoints."""
    return sum(
        math.hypot(b.x - a.x, b.y - a.y)
        for a, b in zip(waypoints, waypoints[1:])
    )
