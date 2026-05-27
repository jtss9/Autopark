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
    metrics: dict = field(default_factory=dict)


def plan_trajectory(
    pc: ParkingConfig,
    cc: CarConfig,
    planner: str = "baseline",
) -> TrajectoryResult:
    if planner == "hybrid_astar":
        from hybrid_astar import plan_hybrid_astar
        return plan_hybrid_astar(pc, cc)

    if pc.obstacle_scenario != "none":
        from hybrid_astar import plan_hybrid_astar
        return plan_hybrid_astar(pc, cc)

    if pc.parking_type == "perpendicular":
        return _plan_perpendicular_mpc(pc, cc)
    if pc.parking_type == "parallel":
        from hybrid_astar import plan_hybrid_astar
        return plan_hybrid_astar(pc, cc)
    return TrajectoryResult([], False, f"Unknown parking type: {pc.parking_type}")


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
