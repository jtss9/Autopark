"""
Parking trajectory planners.
All coordinates use world space: +x right, +y up, units in metres.
The reference point for every Waypoint is the REAR AXLE centre.

Perpendicular maneuver (倒車入庫):
  Geometric planner produces a reference path (3-phase arc).
  MPC planner uses that reference + boundary penalty to generate
  a physically-simulated, boundary-safe trajectory.
"""
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff, path_length as _path_length
from parking_lot import ParkingLot


@dataclass
class Waypoint:
    x: float
    y: float
    theta: float   # heading in radians


@dataclass
class TrajectoryResult:
    waypoints: List[Waypoint]
    feasible: bool
    message: str
    phase_starts: List[int] = field(default_factory=list)
    phase_names:  List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    executed_waypoints: List[Waypoint] = field(default_factory=list)
    tracking_metrics: dict = field(default_factory=dict)


def plan_trajectory(
    pc: ParkingConfig,
    cc: CarConfig,
    planner: Optional[str] = None,
    track: bool = False,
) -> TrajectoryResult:
    """Dispatch to the right planner backend.

    `planner` is the explicit caller intent and always wins. When None,
    we fall back to pc.planner so the Settings-UI selection is honoured.
    The previous `pc.planner == "hybrid_astar"` OR branches silently
    shadowed an explicit baseline request — fixed by making the
    resolution explicit and consulting pc.planner only as fallback.
    """
    effective = planner if planner is not None else pc.planner

    if effective == "qlearn":
        from rl_qlearn import plan_qlearn
        result = plan_qlearn(pc, cc)
    elif effective == "hybrid_astar":
        from hybrid_astar import plan_hybrid_astar
        result = plan_hybrid_astar(pc, cc)
    elif pc.obstacle_scenario != "none":
        # MPC backends have no obstacle awareness; auto-promote to Hybrid A*
        # for any non-empty scenario so we never produce a colliding plan.
        from hybrid_astar import plan_hybrid_astar
        result = plan_hybrid_astar(pc, cc)
    elif pc.parking_type == "perpendicular":
        if effective == "multi":
            result = _plan_perpendicular_multistep(pc, cc)
        else:
            result = _plan_perpendicular_mpc(pc, cc)
    elif pc.parking_type == "parallel":
        from hybrid_astar import plan_hybrid_astar
        result = plan_hybrid_astar(pc, cc)
    else:
        return TrajectoryResult([], False, f"Unknown parking type: {pc.parking_type}")

    if track and result.feasible and len(result.waypoints) >= 2:
        _attach_tracker(pc, cc, result)
    return result


def _attach_tracker(
    pc: ParkingConfig,
    cc: CarConfig,
    result: TrajectoryResult,
) -> None:
    """Run a Pure Pursuit closed-loop tracker on the planned path."""
    from tracker import track_path
    lot = ParkingLot(pc, cc)
    dense = _densify_for_tracking(result.waypoints, max_step=0.15)
    tr = track_path(dense, cc, lot)
    result.executed_waypoints = tr.executed
    result.tracking_metrics = {
        "tracker": "pure_pursuit",
        "tracker_success": tr.succeeded,
        "tracker_message": tr.message,
        "mean_cte_m": tr.mean_cte_m,
        "max_cte_m": tr.max_cte_m,
        "exec_final_pos_error_m": tr.final_pos_error_m,
        "exec_final_heading_error_deg": tr.final_heading_error_deg,
        "exec_fully_in_spot": tr.fully_in_spot,
        "cusps": tr.cusps,
    }


def _densify_for_tracking(
    waypoints: List[Waypoint],
    max_step: float = 0.15,
) -> List[Waypoint]:
    """Insert intermediate samples so consecutive waypoints stay within max_step."""
    if len(waypoints) < 2:
        return list(waypoints)
    out: List[Waypoint] = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        dx = b.x - a.x
        dy = b.y - a.y
        dist = math.hypot(dx, dy)
        n = max(1, int(math.ceil(dist / max_step)))
        for k in range(1, n + 1):
            t = k / n
            theta = a.theta + _angle_diff(b.theta, a.theta) * t
            out.append(Waypoint(a.x + dx * t, a.y + dy * t, theta))
    return out


def _goal_pose(lot: ParkingLot) -> tuple:
    spot = lot.spot_rect
    if lot.pc.parking_type == "parallel":
        return spot.x + 0.15, spot.y + spot.h / 2, 0.0
    return spot.x + spot.w / 2, spot.top - 0.15, -math.pi / 2


def _fully_in_spot(lot: ParkingLot, waypoints: List[Waypoint]) -> bool:
    if not waypoints:
        return False
    spot = lot.spot_rect
    final = waypoints[-1]
    return all(
        spot.x <= x <= spot.right and spot.y <= y <= spot.top
        for x, y in lot.car_corners((final.x, final.y, final.theta))
    )


def _result_with_metrics(
    pc: ParkingConfig,
    cc: CarConfig,
    waypoints: List[Waypoint],
    feasible: bool,
    message: str,
    phase_starts: List[int],
    phase_names: List[str],
    planning_time_s: float,
    iterations: int = 0,
) -> TrajectoryResult:
    lot = ParkingLot(pc, cc)
    goal = _goal_pose(lot)
    fully_in_spot = _fully_in_spot(lot, waypoints)

    final_pos_error = 0.0
    final_heading_error = 0.0
    if waypoints:
        final = waypoints[-1]
        final_pos_error = math.hypot(final.x - goal[0], final.y - goal[1])
        final_heading_error = math.degrees(abs(_angle_diff(final.theta, goal[2])))

    final_feasible = feasible and fully_in_spot
    final_message = message
    if feasible and not fully_in_spot:
        final_message = "Final car is not fully inside the parking spot."

    # NOTE: MPC reports `expanded_states = 0` because it does not perform a
    # graph search. Hybrid A* sets this to the search-tree size and Q-learn
    # sets it to the Q-table size; keeping MPC at 0 makes cross-planner
    # comparisons of search effort unambiguous (rather than overloading the
    # column with waypoint count, which has a separate `waypoints` field).
    metrics = {
        "planning_time_s": planning_time_s,
        "iterations": iterations,
        "expanded_states": 0,
        "path_length_m": _path_length(waypoints),
        "waypoints": len(waypoints),
        "final_pos_error_m": final_pos_error,
        "final_heading_error_deg": final_heading_error,
        "fully_in_spot": fully_in_spot,
        "obstacles": 0,
    }
    return TrajectoryResult(
        waypoints,
        final_feasible,
        final_message,
        phase_starts,
        phase_names,
        metrics,
    )


# ---------------------------------------------------------------------------
# Geometric reference planner (used internally by MPC planner)
# ---------------------------------------------------------------------------
def _plan_perpendicular(pc: ParkingConfig, cc: CarConfig) -> TrajectoryResult:
    lot  = ParkingLot(pc, cc)
    R    = cc.min_turn_radius
    L    = cc.length
    W    = cc.width

    x_start, y_lane, _ = lot.car_start_pose
    x_spot = lot.spot_rect.x + lot.spot_rect.w / 2
    x_stop = x_spot + R

    # ── Feasibility checks (record but don't abort — always generate path) ──
    feasible = True
    message  = "OK"

    if x_stop >= lot.lane_rect.right - 0.1:
        feasible = False
        message  = (f"Not enough lane length to the right of the spot "
                    f"(need {R:.1f} m extra, lane ends at {lot.lane_rect.right:.1f} m).")
        x_stop = lot.lane_rect.right - 0.15   # clamp so path stays drawable

    t_crit      = math.atan2(L, R + W / 2)
    min_front_y = (y_lane
                   + R * (1 - math.cos(t_crit))
                   - L * math.sin(t_crit)
                   - (W / 2) * math.cos(t_crit))
    if min_front_y < 0.0 and feasible:
        needed_lane = pc.lane_width - min_front_y
        feasible = False
        message  = (f"Infeasible: car front clips the road edge during the arc. "
                    f"Try a wider lane (≥ {needed_lane:.1f} m) or a shorter car.")

    if not lot.car_fits() and feasible:
        feasible = False
        message  = "Car is larger than the parking spot."

    # ── Waypoint generation ────────────────────────────────────────────
    STEP    = 0.05
    ARC_PTS = 91

    wps: List[Waypoint] = []
    phase_starts: List[int] = []
    phase_names:  List[str] = []

    phase_starts.append(0)
    phase_names.append("Drive forward")
    x = x_start
    while x < x_stop - STEP / 2:
        wps.append(Waypoint(x, y_lane, 0.0))
        x += STEP
    wps.append(Waypoint(x_stop, y_lane, 0.0))

    phase_starts.append(len(wps))
    phase_names.append("Reversing - arc")
    for i in range(ARC_PTS):
        t  = math.radians(i)
        rx = x_stop - R * math.sin(t)
        ry = y_lane + R * (1.0 - math.cos(t))
        wps.append(Waypoint(rx, ry, -t))
    wps.append(Waypoint(x_spot, y_lane + R, -math.pi / 2))

    phase_starts.append(len(wps))
    phase_names.append("Reversing into spot")
    y_arc_end = y_lane + R
    spot_top  = pc.lane_width + pc.spot_length
    target_y  = max(spot_top - 0.15, y_arc_end)
    y = y_arc_end
    while y < target_y - STEP / 2:
        wps.append(Waypoint(x_spot, y, -math.pi / 2))
        y += STEP
    wps.append(Waypoint(x_spot, target_y, -math.pi / 2))

    return TrajectoryResult(wps, feasible, message, phase_starts, phase_names)


# ---------------------------------------------------------------------------
# MPC planner — kinematic simulation guided by MPC
# ---------------------------------------------------------------------------
def _plan_perpendicular_mpc(pc: ParkingConfig, cc: CarConfig) -> TrajectoryResult:
    from controller import CarDynamics, MPCController

    # Step 1: geometric reference path (always generated; MPC detects actual violations)
    ref = _plan_perpendicular(pc, cc)

    lot    = ParkingLot(pc, cc)
    x_spot = lot.spot_rect.x + lot.spot_rect.w / 2
    x_stop = min(x_spot + cc.min_turn_radius, lot.lane_rect.right - 0.15)

    x0, y0, th0 = lot.car_start_pose
    car = CarDynamics(x0, y0, th0, cc)
    mpc = MPCController(lot, cc, ref.waypoints)

    FORWARD_V = 3.0
    REVERSE_V = 1.5
    dt = mpc.dt

    # Index in the reference where the reversing arc begins
    p1_end = ref.phase_starts[1] if len(ref.phase_starts) > 1 else len(ref.waypoints)

    wps:          List[Waypoint] = [Waypoint(car.x, car.y, car.theta)]
    phase_starts: List[int]      = [0]
    phase_names:  List[str]      = ["Drive forward"]

    t0 = time.perf_counter()
    print("Planning MPC trajectory...", end="", flush=True)

    # ── Phase 1: straight forward to x_stop (delta=0, no MPC needed) ──────
    while car.x < x_stop - 0.05:
        car.step(FORWARD_V, 0.0, dt)
        wps.append(Waypoint(car.x, car.y, car.theta))
        if not mpc.corners_in_bounds(car.x, car.y, car.theta):
            print(f" failed ({time.perf_counter() - t0:.1f}s)")
            return _result_with_metrics(
                pc, cc,
                wps, False, "Car body exceeded valid area during forward drive.",
                phase_starts, phase_names, time.perf_counter() - t0, len(wps))

    # ── Phase 2+3: MPC reverse arc + straight into spot ────────────────────
    phase_starts.append(len(wps))
    phase_names.append("Reversing (MPC)")
    mpc.ref_idx = p1_end   # start tracking from the arc portion of the reference

    goal       = ref.waypoints[-1]
    prev_delta = 0.0
    iterations = 0

    for _ in range(1500):
        iterations += 1
        delta = mpc.optimize(car, -REVERSE_V, prev_delta)
        car.step(-REVERSE_V, delta, dt)
        prev_delta = delta

        if not mpc.corners_in_bounds(car.x, car.y, car.theta):
            print(f" failed ({time.perf_counter() - t0:.1f}s)")
            wps.append(Waypoint(car.x, car.y, car.theta))  # include collision pose
            return _result_with_metrics(
                pc, cc,
                wps, False,
                "Car body exceeded valid area — try a wider lane or shorter car.",
                phase_starts, phase_names, time.perf_counter() - t0, iterations)

        wps.append(Waypoint(car.x, car.y, car.theta))

        dist = math.hypot(car.x - goal.x, car.y - goal.y)
        dh   = abs((car.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi)
        if dist < 0.15 and dh < 0.1:
            break
    else:
        print(f" timed out ({time.perf_counter() - t0:.1f}s)")
        return _result_with_metrics(
            pc, cc,
            wps, False, "MPC: could not reach parking goal within step limit.",
            phase_starts, phase_names, time.perf_counter() - t0, iterations)

    elapsed = time.perf_counter() - t0
    print(f" done in {elapsed:.1f}s ({len(wps)} waypoints)")
    return _result_with_metrics(
        pc, cc, wps, True, "OK", phase_starts, phase_names, elapsed, iterations)


# ---------------------------------------------------------------------------
# Multi-step planner — state machine + goal-directed NMPC
# ---------------------------------------------------------------------------
def _plan_perpendicular_multistep(pc: ParkingConfig, cc: CarConfig) -> TrajectoryResult:
    """
    Multi-step planner.  Each reverse attempt uses a fresh geometric arc
    reference computed from the current (elevated) car position, tracked by the
    same reference-MPC as the single-step planner.  When the warn margin fires,
    a goal-directed correction drives the car back to x_stop while HOLDING the
    y-elevation gained — so each attempt starts higher and sweeps with more
    front-corner clearance.
    """
    from controller import CarDynamics, MPCController

    lot    = ParkingLot(pc, cc)
    x_spot = lot.spot_rect.x + lot.spot_rect.w / 2
    x_stop = min(x_spot + cc.min_turn_radius, lot.lane_rect.right - 0.15)

    FORWARD_V    = 3.0
    REVERSE_V    = 1.5
    MAX_ATTEMPTS = 5
    WARN_MARGIN  = 0.15   # triggers correction when any corner within this of road bottom
    dt           = 0.05

    # Build a fresh geometric arc reference from (x0, y0, theta0).
    # Generates the remaining arc needed to reach theta=-pi/2, so each
    # attempt continues from the car's actual heading rather than assuming 0.
    def _arc_ref(x0: float, y0: float, theta0: float = 0.0) -> List[Waypoint]:
        R    = cc.min_turn_radius
        STEP = 0.05
        # ICR for a left-turn reverse maneuver starting at theta0
        icr_x = x0 - R * math.sin(theta0)
        icr_y = y0 + R * math.cos(theta0)
        # Sweep from theta0 to -pi/2. Clamp to >= 0 so a car already at or
        # past -pi/2 yields a zero-length arc and falls through to the
        # straight reverse below, instead of sweeping backwards on a
        # degenerate 2-point reference.
        arc_span = max(0.0, theta0 + math.pi / 2)
        n_arc = max(2, int(math.degrees(arc_span)) + 1)
        ref: List[Waypoint] = []
        for i in range(n_arc):
            t  = arc_span * i / (n_arc - 1)
            th = theta0 - t
            ref.append(Waypoint(icr_x + R * math.sin(th),
                                icr_y - R * math.cos(th), th))
        x_end = icr_x - R            # x at theta=-pi/2
        y_end = icr_y                # y at theta=-pi/2
        y = y_end + STEP
        spot_top = lot.lane_rect.top + lot.spot_rect.h
        # Reverse straight all the way to the goal depth. The previous
        # `min(..., y_end + 1.5)` cap stopped 1.5 m past the arc exit, which
        # left the car ~3 m short of the goal in a deep spot (the arc exits at
        # y_end ~= y0 + R, well below spot_top for any non-shallow spot).
        target_y = spot_top - 0.15
        while y < target_y - STEP / 2:
            ref.append(Waypoint(x_end, y, -math.pi / 2))
            y += STEP
        ref.append(Waypoint(x_end, target_y, -math.pi / 2))
        return ref

    car = CarDynamics(*lot.car_start_pose, cc)
    bounds_mpc = MPCController(lot, cc, [])   # used only for boundary checks

    wps:          List[Waypoint] = [Waypoint(car.x, car.y, car.theta)]
    phase_starts: List[int]      = [0]
    phase_names:  List[str]      = ["Drive forward"]

    t0 = time.perf_counter()
    print("Planning multi-step MPC trajectory...", end="", flush=True)
    iterations = 0

    # ── Phase 1: straight forward to x_stop ───────────────────────────────────
    while car.x < x_stop - 0.05:
        car.step(FORWARD_V, 0.0, dt)
        wps.append(Waypoint(car.x, car.y, car.theta))
        if not bounds_mpc.corners_in_bounds(car.x, car.y, car.theta):
            print(f" failed ({time.perf_counter() - t0:.1f}s)")
            return _result_with_metrics(
                pc, cc,
                wps, False, "Car body exceeded valid area during forward drive.",
                phase_starts, phase_names, time.perf_counter() - t0, len(wps))

    # ── Phase 2+: alternating REVERSING / FORWARD_CORRECT ─────────────────────
    prev_delta = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Fresh arc reference continuing from the car's current heading
        ref_wps  = _arc_ref(car.x, car.y, car.theta)
        rev_mpc  = MPCController(lot, cc, ref_wps)
        rev_mpc.ref_idx = 0
        goal     = ref_wps[-1]

        phase_starts.append(len(wps))
        phase_names.append(f"Reversing (attempt {attempt})")

        near_boundary = False
        for _ in range(1500):
            iterations += 1
            delta = rev_mpc.optimize(car, -REVERSE_V, prev_delta)
            car.step(-REVERSE_V, delta, dt)
            prev_delta = delta

            if not bounds_mpc.corners_in_bounds(car.x, car.y, car.theta):
                wps.append(Waypoint(car.x, car.y, car.theta))
                print(f" failed ({time.perf_counter() - t0:.1f}s)")
                return _result_with_metrics(
                    pc, cc,
                    wps, False,
                    "Car body exceeded valid area — try a wider lane or shorter car.",
                    phase_starts, phase_names, time.perf_counter() - t0, iterations)

            wps.append(Waypoint(car.x, car.y, car.theta))

            dist = math.hypot(car.x - goal.x, car.y - goal.y)
            dh   = abs((car.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi)
            if dist < 0.15 and dh < 0.1:
                elapsed = time.perf_counter() - t0
                print(f" done in {elapsed:.1f}s ({len(wps)} waypoints, "
                      f"{attempt} attempt(s))")
                return _result_with_metrics(
                    pc, cc, wps, True, "OK", phase_starts, phase_names,
                    elapsed, iterations)

            # Only trigger correction for road-bottom approach — the MPC
            # boundary penalty handles spot/side boundaries.
            corners = lot.car_corners((car.x, car.y, car.theta))
            if any(cy < lot.lane_rect.y + WARN_MARGIN for _, cy in corners):
                near_boundary = True
                break

        if not near_boundary or attempt == MAX_ATTEMPTS:
            break

        # ── Correction (2-phase) ──────────────────────────────────────────────
        # Phase A: reverse with hard RIGHT steer until theta ≈ 0°.
        #   dy/dt = v·sin(theta) = (−v)·sin(negative θ) > 0  → gains y clearance.
        # Phase B: drive FORWARD straight (delta=0) back to x_stop.
        #   At theta ≈ 0°, sin(theta) ≈ 0  → y stays constant while x is restored.
        phase_starts.append(len(wps))
        phase_names.append(f"Forward correction {attempt}")

        # Phase A
        for _ in range(100):
            iterations += 1
            car.step(-REVERSE_V, -cc.max_steer, dt)
            prev_delta = -cc.max_steer
            wps.append(Waypoint(car.x, car.y, car.theta))
            if not bounds_mpc.corners_in_bounds(car.x, car.y, car.theta):
                print(f" failed ({time.perf_counter() - t0:.1f}s)")
                return _result_with_metrics(
                    pc, cc,
                    wps, False, "Car body exceeded valid area during correction.",
                    phase_starts, phase_names, time.perf_counter() - t0, iterations)
            if abs(car.theta) < math.radians(5):
                break

        # Phase B — drive forward to an x position safe for the next arc.
        # With elevated y, going all the way to x_stop would cause the car
        # body to clip the spot's right boundary when entering.  Compute the
        # maximum safe x_start so the body stays within spot.right + margin.
        R_turn   = cc.min_turn_radius
        half_w   = cc.width / 2
        denom    = R_turn - half_w                     # (R - W/2)
        icr_diff = car.y + R_turn - lot.lane_rect.top  # ICR_y - lane_top
        if 0 < icr_diff < denom:
            sin_tc   = math.sqrt(max(0.0, 1.0 - (icr_diff / denom) ** 2))
            x_target = min(x_stop, lot.spot_rect.right + 0.05 + denom * sin_tc - 0.10)
        else:
            x_target = x_stop
        x_target = max(x_target, car.x + 0.30)        # always drive at least 30 cm

        for _ in range(300):
            iterations += 1
            car.step(FORWARD_V, 0.0, dt)
            prev_delta = 0.0
            wps.append(Waypoint(car.x, car.y, car.theta))
            if not bounds_mpc.corners_in_bounds(car.x, car.y, car.theta):
                print(f" failed ({time.perf_counter() - t0:.1f}s)")
                return _result_with_metrics(
                    pc, cc,
                    wps, False, "Car body exceeded valid area during correction.",
                    phase_starts, phase_names, time.perf_counter() - t0, iterations)
            if car.x >= x_target - 0.05:
                break

    print(f" timed out ({time.perf_counter() - t0:.1f}s)")
    return _result_with_metrics(
        pc, cc,
        wps, False,
        f"Could not park within {MAX_ATTEMPTS} attempts.",
        phase_starts, phase_names, time.perf_counter() - t0, iterations)
