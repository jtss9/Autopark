"""
Hybrid A* planner for perpendicular (reverse-into-spot) parking.

State space: continuous (x, y, theta), discretised by a 3-tuple grid key for
visited-set deduplication.  Motion primitives are forward/reverse ×
five steering angles integrated with the bicycle kinematic model.

Public API
----------
    plan_hastar(lot, cc, start, goal) -> (path, feasible, message)

    path     : list of (x, y, theta, direction) — direction +1 fwd / -1 rev
    feasible : True if a path was found
    message  : "OK" or a human-readable failure reason
"""
import math
import heapq
from typing import List, Tuple

from config import CarConfig
from parking_lot import ParkingLot

# ── Tuning parameters ─────────────────────────────────────────────────────────
_GRID_RES       = 0.30           # m — spatial cell size and arc length per primitive
_THETA_RES      = math.pi / 36  # 5° per heading bucket
_N_THETA        = round(2 * math.pi / _THETA_RES)  # 72 buckets
_ARC_SUBSTEPS   = 6             # collision-check sub-steps per primitive
_REVERSE_COST   = 1.5           # g-cost multiplier for reverse moves
_GEAR_PENALTY   = 1.0           # extra g-cost per direction change
_MAX_EXPANSIONS = 80_000        # search node limit
_GOAL_DIST      = 0.30          # m — success position threshold
_GOAL_HEADING   = 0.15          # rad — success heading threshold (~8.6°)
_MARGIN         = 0.05          # m — collision boundary margin


# ── Grid key ──────────────────────────────────────────────────────────────────

def _discretise(x: float, y: float, theta: float) -> Tuple[int, int, int]:
    ix = int(math.floor(x / _GRID_RES))
    iy = int(math.floor(y / _GRID_RES))
    it = int(round(theta / _THETA_RES)) % _N_THETA
    return (ix, iy, it)


# ── Kinematic integration ─────────────────────────────────────────────────────

def _kinematic_step(x: float, y: float, theta: float,
                    direction: int, delta: float,
                    arc_len: float, wheelbase: float,
                    n_sub: int) -> Tuple[float, float, float, list]:
    """
    Integrate bicycle kinematics over arc_len using n_sub equal sub-steps.
    Returns (new_x, new_y, new_theta, list_of_sub_step_poses).
    Each element of list_of_sub_step_poses is (x, y, theta).
    """
    step = arc_len / n_sub
    poses = []
    for _ in range(n_sub):
        x     += direction * step * math.cos(theta)
        y     += direction * step * math.sin(theta)
        theta += direction * step * math.tan(delta) / wheelbase
        poses.append((x, y, theta))
    return x, y, theta, poses


# ── Boundary check ────────────────────────────────────────────────────────────

def _in_bounds(corners, lane, spot) -> bool:
    """
    Valid region = lane_rect ∪ spot_rect  (L-shape).
    Each corner must satisfy:
      • vertical:   lane.y + M  ≤  cy  ≤  spot.top − M
      • horizontal: lane.x + M  ≤  cx  ≤  lane.right − M
      • above lane: cy > lane.top − M  →  also  spot.x + M ≤ cx ≤ spot.right − M
    """
    M  = _MARGIN
    lt = lane.top - M
    for cx, cy in corners:
        if cy < lane.y + M or cy > spot.top - M:
            return False
        if cx < lane.x + M or cx > lane.right - M:
            return False
        if cy > lt:
            if cx < spot.x + M or cx > spot.right - M:
                return False
    return True


# ── Admissible heuristic ──────────────────────────────────────────────────────

def _heuristic(x: float, y: float, theta: float,
               gx: float, gy: float, gtheta: float) -> float:
    dist = math.hypot(x - gx, y - gy)
    dh   = abs((theta - gtheta + math.pi) % (2 * math.pi) - math.pi)
    return dist + 0.5 * dh


# ── Path reconstruction ───────────────────────────────────────────────────────

def _backtrack(came_from: dict, state_pos: dict,
               final_key, start_key) -> List[Tuple[float, float, float, int]]:
    """Follow parent pointers from final_key to start_key, then reverse."""
    segments: List[Tuple[list, int]] = []
    key = final_key
    while came_from[key] is not None:
        parent_key, sub_poses, direction = came_from[key]
        segments.append((sub_poses, direction))
        key = parent_key
    segments.reverse()

    sx, sy, st = state_pos[start_key]
    first_dir  = segments[0][1] if segments else 1
    path: List[Tuple[float, float, float, int]] = [(sx, sy, st, first_dir)]
    for sub_poses, direction in segments:
        for px, py, pt in sub_poses:
            path.append((px, py, pt, direction))
    return path


# ── Main entry point ──────────────────────────────────────────────────────────

def plan_hastar(
    lot: ParkingLot,
    cc: CarConfig,
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
) -> Tuple[List[Tuple[float, float, float, int]], bool, str]:
    """
    Hybrid A* search from *start* to *goal* in the drivable region of *lot*.

    Parameters
    ----------
    lot   : ParkingLot — provides lane_rect, spot_rect, car_corners()
    cc    : CarConfig  — provides wheelbase, max_steer
    start : (x, y, theta) rear-axle pose in world space
    goal  : (x, y, theta) target rear-axle pose

    Returns
    -------
    (path, feasible, message)
    path     : list of (x, y, theta, direction) tuples (+1 fwd, -1 rev)
    feasible : True if a valid path was found
    message  : "OK" or a human-readable failure description
    """
    lane = lot.lane_rect
    spot = lot.spot_rect
    wb   = cc.wheelbase
    ms   = cc.max_steer

    steers = [-ms, -ms * 0.5, 0.0, ms * 0.5, ms]
    dirs   = [1, -1]

    gx, gy, gtheta = goal
    sx, sy, stheta = start
    start_key = _discretise(sx, sy, stheta)

    # came_from[key] = None (start) | (parent_key, sub_poses, direction)
    # state_pos[key] = best known (x, y, theta) at that grid key
    came_from: dict = {start_key: None}
    state_pos: dict = {start_key: (sx, sy, stheta)}
    g_cost: dict    = {start_key: 0.0}

    h0 = _heuristic(sx, sy, stheta, gx, gy, gtheta)
    # heap: (f, g, tie_breaker, x, y, theta, last_direction)
    counter = 0
    heap = [(h0, 0.0, counter, sx, sy, stheta, 0)]

    n_expansions = 0

    while heap:
        f, g, _, x, y, theta, last_dir = heapq.heappop(heap)
        n_expansions += 1

        if n_expansions > _MAX_EXPANSIONS:
            return [], False, (
                f"Hybrid A* reached search limit ({_MAX_EXPANSIONS:,} nodes). "
                "The scene may be geometrically infeasible for this car/lane combination."
            )

        cur_key = _discretise(x, y, theta)

        # Stale heap entry — a cheaper path to this cell exists
        if g > g_cost.get(cur_key, float('inf')) + 1e-6:
            continue

        # ── Goal check ────────────────────────────────────────────────────
        dist = math.hypot(x - gx, y - gy)
        dh   = abs((theta - gtheta + math.pi) % (2 * math.pi) - math.pi)
        if dist < _GOAL_DIST and dh < _GOAL_HEADING:
            return _backtrack(came_from, state_pos, cur_key, start_key), True, "OK"

        # ── Expand neighbours ─────────────────────────────────────────────
        for direction in dirs:
            for delta in steers:
                nx, ny, ntheta, sub_poses = _kinematic_step(
                    x, y, theta, direction, delta,
                    _GRID_RES, wb, _ARC_SUBSTEPS,
                )
                # Collision-check every sub-step
                valid = True
                for px, py, pt in sub_poses:
                    if not _in_bounds(lot.car_corners((px, py, pt)), lane, spot):
                        valid = False
                        break
                if not valid:
                    continue

                move_cost = _GRID_RES * (_REVERSE_COST if direction == -1 else 1.0)
                if last_dir != 0 and direction != last_dir:
                    move_cost += _GEAR_PENALTY
                ng = g + move_cost

                nkey = _discretise(nx, ny, ntheta)
                if ng < g_cost.get(nkey, float('inf')):
                    g_cost[nkey]    = ng
                    state_pos[nkey] = (nx, ny, ntheta)
                    came_from[nkey] = (cur_key, sub_poses, direction)
                    h = _heuristic(nx, ny, ntheta, gx, gy, gtheta)
                    counter += 1
                    heapq.heappush(heap, (ng + h, ng, counter,
                                          nx, ny, ntheta, direction))

    return [], False, "No path found — target pose unreachable with current car/lane geometry."
