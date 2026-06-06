"""
Reeds-Shepp shortest paths between two SE(2) poses.

Reference: Reeds & Shepp, "Optimal paths for a car that goes both forwards and
backwards", Pacific Journal of Mathematics, 1990.

This module implements the standard 12-word Reeds-Shepp family (CSC + CCC),
combined with the time-flip symmetry to cover the full reverse motion set.
That gives 24 candidate words, which already covers the vast majority of
practical parking maneuvers. It is used by `hybrid_astar.py` as an analytic
"shot" so that the search can terminate with a smooth, vehicle-feasible
maneuver once it gets near the goal.

Conventions:
  - Pose is (x, y, theta), theta in radians.
  - The car has unit turning radius internally; the API takes a `radius`
    parameter and rescales transparently.
  - A path is a list of segments (length, steer, gear):
      steer in {-1, 0, 1}  -> right, straight, left
      gear  in {-1, +1}    -> reverse, forward
      length >= 0          -> arc length (or straight length) in metres
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

PI = math.pi
TWO_PI = 2.0 * PI


@dataclass(frozen=True)
class Segment:
    length: float   # metres
    steer: int      # -1 right, 0 straight, +1 left
    gear: int       # +1 forward, -1 reverse


@dataclass
class RSPath:
    segments: List[Segment]

    @property
    def length(self) -> float:
        return sum(s.length for s in self.segments)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def _mod2pi(theta: float) -> float:
    v = theta % TWO_PI
    if v < 0:
        v += TWO_PI
    return v


def _pi_to_pi(theta: float) -> float:
    return (theta + PI) % TWO_PI - PI


def _polar(x: float, y: float) -> Tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


# ---------------------------------------------------------------------------
# Base Reeds-Shepp formulas (unit-radius, all on canonical frame)
# Each returns (t, u, v) lengths >= 0 or None if infeasible.
# These formulas follow the standard Reeds-Shepp derivation.
# ---------------------------------------------------------------------------
def _LpSpLp(x: float, y: float, phi: float):
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t < -1e-9:
        return None
    v = _mod2pi(phi - t)
    return t, u, v


def _LpSpRp(x: float, y: float, phi: float):
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1_sq = u1 * u1
    if u1_sq < 4.0:
        return None
    u = math.sqrt(u1_sq - 4.0)
    theta = math.atan2(2.0, u)
    t = _mod2pi(t1 + theta)
    v = _mod2pi(t - phi)
    return t, u, v


def _LpRnLp(x: float, y: float, phi: float):
    """C|C|C with first/last forward-left, middle reverse-right."""
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 > 4.0:
        return None
    A = math.acos(min(1.0, u1 / 4.0))
    t = _mod2pi(theta + PI / 2 + A)
    u = _mod2pi(PI - 2.0 * A)
    v = _mod2pi(phi - t - u)
    return t, u, v


def _LpRnLn(x: float, y: float, phi: float):
    """C|CC with last segment reversed direction."""
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 > 4.0:
        return None
    A = math.acos(min(1.0, u1 / 4.0))
    t = _mod2pi(theta + PI / 2 + A)
    u = _mod2pi(PI - 2.0 * A)
    v = _mod2pi(t + u - phi)
    return t, u, v


# ---------------------------------------------------------------------------
# Word builders: each turns a base (t,u,v) into a path list of Segments
# applying the proper steer/gear pattern. Steer +1 = L, -1 = R, 0 = S.
# Gear +1 = forward, -1 = reverse.
# ---------------------------------------------------------------------------
def _word_CSC(t, u, v, steer_pattern, gear) -> List[Segment]:
    s1, s2, s3 = steer_pattern
    return [
        Segment(abs(t), s1, gear),
        Segment(abs(u), s2, gear),
        Segment(abs(v), s3, gear),
    ]


def _word_CCC(t, u, v, steer_pattern, gear_pattern) -> List[Segment]:
    s1, s2, s3 = steer_pattern
    g1, g2, g3 = gear_pattern
    return [
        Segment(abs(t), s1, g1),
        Segment(abs(u), s2, g2),
        Segment(abs(v), s3, g3),
    ]


def _all_paths(x: float, y: float, phi: float) -> List[RSPath]:
    """
    Enumerate the 12-word base set + the time-flipped (reverse) variants.
    Each base word generated under the four symmetry combinations:
      identity, timeflip, reflect, timeflip+reflect.
    """
    paths: List[RSPath] = []

    # --- LpSpLp family (forward, then mirrored / time-flipped variants) ---
    for sym in range(4):
        # sym bit 0: timeflip (negates direction & swaps t,u,v sign role)
        # sym bit 1: reflect (mirror about x-axis: y,phi → -y,-phi, L↔R)
        tflip = bool(sym & 1)
        refl = bool(sym & 2)
        xs, ys, ps = x, y, phi
        if tflip:
            xs, ys, ps = -xs, ys, -ps  # backwards-time canonical
        if refl:
            xs, ys, ps = xs, -ys, -ps
        gear = -1 if tflip else 1
        steer_lsl = (-1, 0, -1) if refl else (1, 0, 1)
        steer_lsr = (-1, 0, 1) if refl else (1, 0, -1)

        out = _LpSpLp(xs, ys, ps)
        if out:
            t, u, v = out
            paths.append(RSPath(_word_CSC(t, u, v, steer_lsl, gear)))

        out = _LpSpRp(xs, ys, ps)
        if out:
            t, u, v = out
            paths.append(RSPath(_word_CSC(t, u, v, steer_lsr, gear)))

    # --- CCC family (LpRnLp = L|R|L; build R|L|R via reflect; time-flip too) ---
    for sym in range(4):
        tflip = bool(sym & 1)
        refl = bool(sym & 2)
        xs, ys, ps = x, y, phi
        if tflip:
            xs, ys, ps = -xs, ys, -ps
        if refl:
            xs, ys, ps = xs, -ys, -ps

        # Pattern LpRnLp: first L forward, second R reverse, third L forward.
        # After timeflip everything reverses; after reflect L↔R.
        if not refl:
            steer = (1, -1, 1)   # L R L
        else:
            steer = (-1, 1, -1)  # R L R
        if not tflip:
            gear_pat = (1, -1, 1)
        else:
            gear_pat = (-1, 1, -1)

        out = _LpRnLp(xs, ys, ps)
        if out:
            t, u, v = out
            paths.append(RSPath(_word_CCC(t, u, v, steer, gear_pat)))

        # LpRnLn variant: third segment direction flipped relative to first.
        if not tflip:
            gear_pat2 = (1, -1, -1)
        else:
            gear_pat2 = (-1, 1, 1)
        out = _LpRnLn(xs, ys, ps)
        if out:
            t, u, v = out
            paths.append(RSPath(_word_CCC(t, u, v, steer, gear_pat2)))

    # Filter out degenerate (zero-length) and bogus paths
    cleaned = []
    for p in paths:
        if any(math.isnan(s.length) or math.isinf(s.length) for s in p.segments):
            continue
        if p.length < 1e-6:
            continue
        cleaned.append(p)
    return cleaned


def shortest_path(
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    radius: float,
) -> Optional[RSPath]:
    """Return the shortest Reeds-Shepp path between two world-space poses."""
    if radius <= 0:
        raise ValueError("radius must be positive")

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    cos_s = math.cos(start[2])
    sin_s = math.sin(start[2])

    # Transform goal into start's frame, then scale by 1/radius
    x = (cos_s * dx + sin_s * dy) / radius
    y = (-sin_s * dx + cos_s * dy) / radius
    phi = _pi_to_pi(goal[2] - start[2])

    paths = _all_paths(x, y, phi)
    if not paths:
        return None

    best = min(paths, key=lambda p: p.length)
    return RSPath([Segment(s.length * radius, s.steer, s.gear) for s in best.segments])


# ---------------------------------------------------------------------------
# Discretisation: turn a path into a list of poses
# ---------------------------------------------------------------------------
def discretize(
    start: Tuple[float, float, float],
    path: RSPath,
    radius: float,
    step: float = 0.1,
) -> List[Tuple[float, float, float, int]]:
    """Yield (x, y, theta, gear) samples along the path at ~`step` metres."""
    poses: List[Tuple[float, float, float, int]] = []
    x, y, theta = start
    poses.append((x, y, theta, 0))

    for seg in path.segments:
        if seg.length < 1e-9:
            continue
        n = max(2, int(math.ceil(seg.length / step)))
        ds = seg.length / n
        for _ in range(n):
            if seg.steer == 0:
                x += seg.gear * ds * math.cos(theta)
                y += seg.gear * ds * math.sin(theta)
            else:
                dtheta = seg.gear * seg.steer * ds / radius
                # Use small-step circular integration for numerical stability
                x += seg.gear * ds * math.cos(theta + 0.5 * dtheta)
                y += seg.gear * ds * math.sin(theta + 0.5 * dtheta)
                theta = _pi_to_pi(theta + dtheta)
            poses.append((x, y, theta, seg.gear))
    return poses


def path_length(start, goal, radius) -> float:
    """Lower-bound non-holonomic heuristic length (returns +inf if no RS path)."""
    p = shortest_path(start, goal, radius)
    return p.length if p else float("inf")
