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
from typing import List

from config import CarConfig, ParkingConfig
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


def plan_trajectory(pc: ParkingConfig, cc: CarConfig) -> TrajectoryResult:
    if pc.parking_type == "perpendicular":
        if pc.planner == "hastar":
            return _plan_perpendicular_hastar(pc, cc)
        if pc.planner == "multi":
            return _plan_perpendicular_multistep(pc, cc)
        return _plan_perpendicular_mpc(pc, cc)
    return TrajectoryResult([], False, "Parallel parking not yet implemented.")


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
            return TrajectoryResult(
                wps, False, "Car body exceeded valid area during forward drive.",
                phase_starts, phase_names)

    # ── Phase 2+3: MPC reverse arc + straight into spot ────────────────────
    phase_starts.append(len(wps))
    phase_names.append("Reversing (MPC)")
    mpc.ref_idx = p1_end   # start tracking from the arc portion of the reference

    goal       = ref.waypoints[-1]
    prev_delta = 0.0

    for _ in range(1500):
        delta = mpc.optimize(car, -REVERSE_V, prev_delta)
        car.step(-REVERSE_V, delta, dt)
        prev_delta = delta

        if not mpc.corners_in_bounds(car.x, car.y, car.theta):
            print(f" failed ({time.perf_counter() - t0:.1f}s)")
            wps.append(Waypoint(car.x, car.y, car.theta))  # include collision pose
            return TrajectoryResult(
                wps, False,
                "Car body exceeded valid area — try a wider lane or shorter car.",
                phase_starts, phase_names)

        wps.append(Waypoint(car.x, car.y, car.theta))

        dist = math.hypot(car.x - goal.x, car.y - goal.y)
        dh   = abs((car.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi)
        if dist < 0.15 and dh < 0.1:
            break
    else:
        print(f" timed out ({time.perf_counter() - t0:.1f}s)")
        return TrajectoryResult(
            wps, False, "MPC: could not reach parking goal within step limit.",
            phase_starts, phase_names)

    elapsed = time.perf_counter() - t0
    print(f" done in {elapsed:.1f}s ({len(wps)} waypoints)")
    return TrajectoryResult(wps, True, "OK", phase_starts, phase_names)


# ---------------------------------------------------------------------------
# Multi-step planner — geometric pre-check + reference MPC
# ---------------------------------------------------------------------------
def _arc_check(x0: float, y0: float, theta0: float,
               cc: CarConfig, lot: ParkingLot,
               margin: float = 0.08) -> bool:
    """True if the geometric arc (theta0 → -pi/2) keeps all car corners
    within every boundary (lane bottom/top, left/right, spot x-range).

    Uses a safety margin so MPC has room to deviate slightly from the arc.
    """
    arc_span = theta0 + math.pi / 2
    if arc_span < math.radians(1):
        return True
    R    = cc.min_turn_radius
    L    = cc.length
    W    = cc.width
    icr_x = x0 - R * math.sin(theta0)
    icr_y = y0 + R * math.cos(theta0)
    lane  = lot.lane_rect
    spot  = lot.spot_rect
    n = max(180, int(math.degrees(arc_span) * 4))
    for i in range(n):
        t  = arc_span * i / (n - 1)
        th = theta0 - t
        cx = icr_x + R * math.sin(th)
        cy = icr_y - R * math.cos(th)
        cos_t, sin_t = math.cos(th), math.sin(th)
        for dx, dy in ((0.0, W/2), (0.0, -W/2), (L, W/2), (L, -W/2)):
            fx = cx + dx * cos_t - dy * sin_t
            fy = cy + dx * sin_t + dy * cos_t
            if fy < lane.y - margin:               return False
            if fy > spot.top + margin:              return False
            if fx < lane.x - margin:               return False
            if fx > lane.right + margin:            return False
            if fy > lane.top + margin:
                if fx < spot.x - margin:           return False
                if fx > spot.right + margin:        return False
    return True


def _plan_perpendicular_multistep(pc: ParkingConfig, cc: CarConfig) -> TrajectoryResult:
    """
    Multi-step planner.

    Each attempt runs MPC tracking a fresh geometric arc reference.
    A WARN fires mid-arc when _arc_check reports the REMAINING arc is infeasible
    (any car corner would clip a boundary).  This correctly handles both the lane-
    bottom issue (narrow lane) and the spot-right-edge issue that arises after
    y-correction.  Wide lanes where single-step succeeds never trigger the WARN.

    After a WARN:
      Phase A — reverse with full right steer until |theta| < 2°  (gains y-clearance).
      Phase B — binary-search for the largest x where _arc_check passes, drive there.
    The elevated y ensures the next attempt's arc stays within all boundaries.
    """
    from controller import CarDynamics, MPCController

    lot    = ParkingLot(pc, cc)
    x_spot = lot.spot_rect.x + lot.spot_rect.w / 2
    x_stop = min(x_spot + cc.min_turn_radius, lot.lane_rect.right - 0.15)

    FORWARD_V        = 3.0
    REVERSE_V        = 1.5
    MAX_ATTEMPTS     = 5
    PARK_Y           = lot.spot_rect.top - 0.20
    dt               = 0.05
    WARN_ENTRY_THETA = -math.radians(25)   # only check after >25° of arc progress

    def _arc_ref(x0: float, y0: float, theta0: float = 0.0) -> List[Waypoint]:
        R    = cc.min_turn_radius
        STEP = 0.05
        icr_x = x0 - R * math.sin(theta0)
        icr_y = y0 + R * math.cos(theta0)
        arc_span = theta0 + math.pi / 2
        n_arc = max(2, int(math.degrees(arc_span)) + 1)
        ref: List[Waypoint] = []
        for i in range(n_arc):
            t  = arc_span * i / (n_arc - 1)
            th = theta0 - t
            ref.append(Waypoint(icr_x + R * math.sin(th),
                                icr_y - R * math.cos(th), th))
        x_end    = icr_x - R
        target_y = lot.spot_rect.top - 0.20
        y = icr_y + STEP
        while y < target_y - STEP / 2:
            ref.append(Waypoint(x_end, y, -math.pi / 2))
            y += STEP
        ref.append(Waypoint(x_end, target_y, -math.pi / 2))
        return ref

    car        = CarDynamics(*lot.car_start_pose, cc)
    bounds_mpc = MPCController(lot, cc, [])

    wps:          List[Waypoint] = [Waypoint(car.x, car.y, car.theta)]
    phase_starts: List[int]      = [0]
    phase_names:  List[str]      = ["Drive forward"]

    t0 = time.perf_counter()
    print("Planning multi-step MPC trajectory...", end="", flush=True)

    # ── Phase 1: straight forward to x_stop ───────────────────────────────────
    while car.x < x_stop - 0.05:
        car.step(FORWARD_V, 0.0, dt)
        wps.append(Waypoint(car.x, car.y, car.theta))
        if not bounds_mpc.corners_in_bounds(car.x, car.y, car.theta):
            print(f" failed ({time.perf_counter() - t0:.1f}s)")
            return TrajectoryResult(
                wps, False, "Car body exceeded valid area during forward drive.",
                phase_starts, phase_names)

    # ── Phase 2+: MPC attempts with arc-feasibility-based corrections ──────────
    prev_delta    = 0.0
    attempt_count = 0

    for correction_num in range(MAX_ATTEMPTS):
        attempt_count += 1
        ref_wps = _arc_ref(car.x, car.y, car.theta)
        rev_mpc = MPCController(lot, cc, ref_wps)
        rev_mpc.ref_idx = 0

        phase_starts.append(len(wps))
        phase_names.append(f"Reversing (attempt {attempt_count})")

        near_boundary = False

        for _ in range(2000):
            delta = rev_mpc.optimize(car, -REVERSE_V, prev_delta)
            car.step(-REVERSE_V, delta, dt)
            prev_delta = delta

            if not bounds_mpc.corners_in_bounds(car.x, car.y, car.theta):
                wps.append(Waypoint(car.x, car.y, car.theta))
                print(f" failed ({time.perf_counter() - t0:.1f}s)")
                return TrajectoryResult(
                    wps, False,
                    "Car body exceeded valid area — try a wider lane or shorter car.",
                    phase_starts, phase_names)

            wps.append(Waypoint(car.x, car.y, car.theta))

            if car.y >= PARK_Y - 0.15:
                elapsed = time.perf_counter() - t0
                print(f" done in {elapsed:.1f}s ({len(wps)} waypoints, "
                      f"{correction_num} correction(s))")
                return TrajectoryResult(wps, True, "OK", phase_starts, phase_names)

            # WARN: only after >25° of arc, check if remaining arc is fully feasible.
            # _arc_check tests ALL boundaries (lane bottom, spot right edge, etc.)
            # so it catches problems that _arc_corner_min_y missed.
            if car.theta < WARN_ENTRY_THETA:
                if not _arc_check(car.x, car.y, car.theta, cc, lot):
                    near_boundary = True
                    break
        else:
            break

        if not near_boundary:
            break

        if correction_num == MAX_ATTEMPTS - 1:
            break

        # ── Forward correction ─────────────────────────────────────────────────
        phase_starts.append(len(wps))
        phase_names.append(f"Forward correction {correction_num + 1}")

        # Phase A: reverse + full right steer until |theta| < 2°
        for _ in range(300):
            car.step(-REVERSE_V, -cc.max_steer, dt)
            prev_delta = -cc.max_steer
            wps.append(Waypoint(car.x, car.y, car.theta))
            if not bounds_mpc.corners_in_bounds(car.x, car.y, car.theta):
                print(f" failed ({time.perf_counter() - t0:.1f}s)")
                return TrajectoryResult(
                    wps, False, "Car body exceeded valid area during Phase A.",
                    phase_starts, phase_names)
            if abs(car.theta) < math.radians(2):
                break

        # Phase B: scan from x_stop downward to find the largest feasible x.
        # Feasibility is non-monotone (too small x → arc parks outside spot;
        # too large x → corner clips spot.right), so binary search fails.
        # Scanning from x_stop down picks the first (largest) feasible x.
        # theta stays constant (delta=0), y changes as Δy = tan(theta)·Δx.
        x_B, y_B, th_B = car.x, car.y, car.theta
        tan_th  = math.tan(th_B)
        x_target = x_B   # fallback: stay put
        SCAN_STEP = 0.05
        x = x_stop
        while x >= x_B - SCAN_STEP / 2:
            y = y_B + (x - x_B) * tan_th
            if _arc_check(x, y, th_B, cc, lot):
                x_target = x
                break
            x -= SCAN_STEP

        for _ in range(500):
            car.step(FORWARD_V, 0.0, dt)
            prev_delta = 0.0
            wps.append(Waypoint(car.x, car.y, car.theta))
            if not bounds_mpc.corners_in_bounds(car.x, car.y, car.theta):
                print(f" failed ({time.perf_counter() - t0:.1f}s)")
                return TrajectoryResult(
                    wps, False, "Car body exceeded valid area during Phase B.",
                    phase_starts, phase_names)
            if car.x >= x_target - 0.05:
                break

    elapsed = time.perf_counter() - t0
    print(f" failed ({elapsed:.1f}s)")
    return TrajectoryResult(
        wps, False,
        f"Could not park within {attempt_count} attempt(s).",
        phase_starts, phase_names)


# ---------------------------------------------------------------------------
# Hybrid A* planner — state-lattice search, no training required
# ---------------------------------------------------------------------------
def _plan_perpendicular_hastar(pc: ParkingConfig, cc: CarConfig) -> TrajectoryResult:
    from hybrid_astar import plan_hastar

    lot = ParkingLot(pc, cc)
    sr  = lot.spot_rect

    start = lot.car_start_pose
    goal  = (sr.x + sr.w / 2,
             sr.top - 0.2,
             -math.pi / 2)

    print("Running Hybrid A*...", end="", flush=True)
    t0 = time.perf_counter()
    raw_path, feasible, message = plan_hastar(lot, cc, start, goal)
    elapsed = time.perf_counter() - t0

    if not feasible:
        print(f" failed ({elapsed:.2f}s)")
        return TrajectoryResult([], False, message)

    waypoints = [Waypoint(x, y, theta) for x, y, theta, _ in raw_path]

    # Build phase labels from direction changes
    phase_starts: List[int] = [0]
    phase_names:  List[str] = ["Forward" if raw_path[0][3] == 1 else "Reverse"]
    for i in range(1, len(raw_path)):
        prev_d = raw_path[i - 1][3]
        curr_d = raw_path[i][3]
        if curr_d != prev_d:
            phase_starts.append(i)
            phase_names.append("Forward" if curr_d == 1 else "Reverse")

    print(f" done ({elapsed:.2f}s, {len(waypoints)} waypoints, "
          f"{len(phase_starts)} phase(s))")
    return TrajectoryResult(
        waypoints, True,
        f"Hybrid A* OK ({elapsed:.2f}s)",
        phase_starts, phase_names,
    )
