"""
End-to-end CARLA autonomous-parking demo.

Pipeline:
    1. Connect to a CARLA server and pick a parking spot (transform).
    2. Spawn the ego vehicle at a "start" location near the spot.
    3. Extract nearby obstacles into our planner frame.
    4. Plan a parking path with Hybrid A* + Reeds-Shepp analytic shot.
    5. Execute the path via Pure Pursuit -> carla.VehicleControl.
    6. Report planned vs executed metrics.

Run modes:
    --carla        Connect to a running CARLA server and run on it.
    --dry-run      Simulate the whole pipeline against our internal kinematic
                   bicycle model — verifies the bridge/controller wiring
                   without requiring CARLA to be installed or running.

The --dry-run path is what we use for CI and for the verification done in
this commit. The --carla path is the production path used on a workstation
that actually has CARLA + a running server.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from carla_bridge import (
    CARLA_AVAILABLE,
    CarlaConnection,
    LocalFrame,
    car_config_from_vehicle,
    extract_static_obstacles,
    pose_from_actor,
    require_carla,
    spawn_ego,
)
from carla_controller import (
    CarlaPurePursuitController,
    ControlCommand,
    ControlConfig,
    control_to_carla,
)
from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff, closest_index as _closest_index
from hybrid_astar import (
    HybridAStarPlanner,
    OccupancyGrid,
    parallel_goal_pose,
    perpendicular_goal_pose,
)
from parking_lot import ParkingLot, Rect
from scenarios import obstacles_for
from trajectory import Waypoint


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class DemoResult:
    planner_success: bool
    planner_message: str
    planning_time_s: float
    planned_waypoints: List[Waypoint]
    executed_poses: List[Tuple[float, float, float]]
    executed_success: bool
    executed_message: str
    final_pos_error_m: float
    final_heading_error_deg: float
    mean_cte_m: float
    max_cte_m: float

    def summary(self) -> str:
        return (
            f"planner_ok={self.planner_success} planning_time={self.planning_time_s:.2f}s "
            f"wp={len(self.planned_waypoints)} | "
            f"executed_ok={self.executed_success} steps={len(self.executed_poses)} "
            f"final_err={self.final_pos_error_m:.3f}m "
            f"heading_err={self.final_heading_error_deg:.2f}deg "
            f"mean_cte={self.mean_cte_m:.3f}m max_cte={self.max_cte_m:.3f}m"
        )


# ---------------------------------------------------------------------------
# Pipeline helpers (shared between --carla and --dry-run)
# ---------------------------------------------------------------------------
def plan_parking(
    parking_type: str,
    cc: CarConfig,
    pc: ParkingConfig,
    obstacles: Sequence[Rect],
) -> Tuple[List[Waypoint], dict]:
    """Run Hybrid A* + RS in the planner frame and return (waypoints, metrics)."""
    lot = ParkingLot(pc, cc)
    grid = OccupancyGrid(lot, obstacles=list(obstacles))
    planner = HybridAStarPlanner(lot, cc, grid)
    if parking_type == "perpendicular":
        goal = perpendicular_goal_pose(lot)
    else:
        goal = parallel_goal_pose(lot)
    start = lot.car_start_pose
    result = planner.plan(start, goal)
    return result.waypoints, result.metrics


def _cte(planned: Sequence[Waypoint], x: float, y: float) -> float:
    i = _closest_index(planned, x, y)
    if i + 1 >= len(planned):
        return math.hypot(x - planned[i].x, y - planned[i].y)
    a, b = planned[i], planned[i + 1]
    seg_len = math.hypot(b.x - a.x, b.y - a.y)
    if seg_len < 1e-6:
        return math.hypot(x - a.x, y - a.y)
    return abs((b.x - a.x) * (a.y - y) - (a.x - x) * (b.y - a.y)) / seg_len


# ---------------------------------------------------------------------------
# --dry-run executor: integrates the controller against a kinematic bicycle
# ---------------------------------------------------------------------------
def _execute_drv_run(
    planned: Sequence[Waypoint],
    cc: CarConfig,
    ctl_cfg: Optional[ControlConfig] = None,
    dt: float = 0.05,
    max_steps: int = 4000,
) -> Tuple[List[Tuple[float, float, float]], List[float], bool, str]:
    """Simulate the controller against our own bicycle model (no CARLA).

    The dry-run integrator constants (accel_per_throttle, decel_per_brake,
    rolling_decel) are intentionally proportional to the controller's
    throttle_kp / brake_kp so the closed loop stays well-behaved when the
    operator retunes the gains via --ctl-* flags.
    """
    ctl = CarlaPurePursuitController(planned, cc, ctl_cfg)

    x, y, theta = planned[0].x, planned[0].y, planned[0].theta
    speed = 0.0  # scalar magnitude
    poses: List[Tuple[float, float, float]] = [(x, y, theta)]
    ctes: List[float] = []

    # Simplified longitudinal model. The accel/decel coefficients are scaled
    # so a controller running at full throttle reaches the configured target
    # speed in ~1.5 s — this keeps the dry-run representative when the
    # operator overrides ControlConfig.target_forward_speed via --ctl-*.
    target_speed = ctl.cfg.target_forward_speed
    accel_per_throttle = max(1.0, target_speed / 0.6)  # m/s^2 at full throttle
    decel_per_brake = max(2.0, 2.0 * accel_per_throttle)
    rolling_decel = 0.4

    for step in range(max_steps):
        cmd: ControlCommand = ctl.step((x, y, theta), speed, dt)
        if cmd.done:
            return poses, ctes, True, "dry-run: controller reported done"

        # Longitudinal
        accel = cmd.throttle * accel_per_throttle - cmd.brake * decel_per_brake - rolling_decel
        speed = max(0.0, speed + accel * dt)
        speed_signed = -speed if cmd.reverse else speed

        # Steering: cmd.steer is normalised; convert to physical delta
        delta = cmd.steer * cc.max_steer
        delta = max(-cc.max_steer, min(cc.max_steer, delta))

        # Bicycle-model step
        x += speed_signed * math.cos(theta) * dt
        y += speed_signed * math.sin(theta) * dt
        theta = _angle_diff(theta + speed_signed * math.tan(delta) / cc.wheelbase * dt, 0.0)

        poses.append((x, y, theta))
        ctes.append(_cte(planned, x, y))

    return poses, ctes, False, f"dry-run: controller did not finish in {max_steps} steps"


def run_dry(
    parking_type: str = "perpendicular",
    obstacle_scenario: str = "none",
    ctl_cfg: Optional[ControlConfig] = None,
) -> DemoResult:
    """Run the full pipeline against the internal simulator (no CARLA)."""
    cc = CarConfig()
    pc = ParkingConfig(parking_type=parking_type, obstacle_scenario=obstacle_scenario)
    lot = ParkingLot(pc, cc)
    obstacles = obstacles_for(lot)

    t0 = time.perf_counter()
    waypoints, planner_metrics = plan_parking(parking_type, cc, pc, obstacles)
    planning_time = time.perf_counter() - t0

    if len(waypoints) < 2:
        return DemoResult(
            planner_success=False,
            planner_message="dry-run: planner returned no usable path",
            planning_time_s=planning_time,
            planned_waypoints=[],
            executed_poses=[],
            executed_success=False,
            executed_message="skipped (no path)",
            final_pos_error_m=float("nan"),
            final_heading_error_deg=float("nan"),
            mean_cte_m=float("nan"),
            max_cte_m=float("nan"),
        )

    poses, ctes, exec_ok, exec_msg = _execute_drv_run(waypoints, cc, ctl_cfg=ctl_cfg)
    fx, fy, fth = poses[-1]
    goal = waypoints[-1]
    pos_err = math.hypot(fx - goal.x, fy - goal.y)
    head_err = math.degrees(abs(_angle_diff(fth, goal.theta)))
    mean_cte = sum(ctes) / len(ctes) if ctes else 0.0
    max_cte = max(ctes) if ctes else 0.0

    return DemoResult(
        planner_success=True,
        planner_message="planner OK",
        planning_time_s=planning_time,
        planned_waypoints=list(waypoints),
        executed_poses=poses,
        executed_success=exec_ok and pos_err < 0.5,
        executed_message=exec_msg,
        final_pos_error_m=pos_err,
        final_heading_error_deg=head_err,
        mean_cte_m=mean_cte,
        max_cte_m=max_cte,
    )


# ---------------------------------------------------------------------------
# --carla executor: actually drives the ego in a running CARLA server
# ---------------------------------------------------------------------------
def run_carla(
    host: str,
    port: int,
    town: Optional[str],
    parking_type: str,
    spot_offset_xy: Tuple[float, float] = (12.0, 0.0),
    max_seconds: float = 60.0,
    ctl_cfg: Optional[ControlConfig] = None,
) -> DemoResult:
    """
    Run the parking pipeline on a real CARLA server.

    The spot is taken to be (0, 0) in the planner frame; the ego is spawned
    at `spot_offset_xy` relative to the spot in the planner frame. In a real
    deployment these come from a parking-lot annotation or a SLAM map; here
    we expose them as simple CLI parameters for the demo. `ctl_cfg` lets
    callers retune the Pure Pursuit gains for a specific vehicle without
    editing source.
    """
    require_carla()
    import carla  # noqa: F401  -- ensure imported

    cc = CarConfig()
    pc = ParkingConfig(parking_type=parking_type)
    lot = ParkingLot(pc, cc)

    # Compute the world-frame spot transform: pick the first parking-spot
    # waypoint of the chosen map. If the map has no parking semantic tag,
    # fall back to the spectator location, which the operator should set
    # before launching the demo.
    with CarlaConnection(host=host, port=port, town=town) as conn:
        world = conn.world
        spectator = world.get_spectator().get_transform()
        spot_xy_world = (spectator.location.x, spectator.location.y)
        spot_yaw_deg = spectator.rotation.yaw
        frame = LocalFrame(origin_xy=spot_xy_world, yaw_offset_deg=-spot_yaw_deg)

        ego_local_x, ego_local_y = spot_offset_xy
        ego_world_x, ego_world_y = frame.local_to_world_xy(ego_local_x, ego_local_y)
        ego_yaw_deg = frame.local_theta_to_world_yaw_deg(0.0)
        spawn_tf = carla.Transform(
            carla.Location(x=ego_world_x, y=ego_world_y, z=spectator.location.z + 0.3),
            carla.Rotation(yaw=ego_yaw_deg),
        )

        ego = spawn_ego(world, spawn_tf)
        try:
            # Override CarConfig with the real vehicle dimensions.
            length, width, wheelbase = car_config_from_vehicle(ego)
            cc = CarConfig(length=length, width=width)
            cc.wheelbase = wheelbase

            # Extract obstacles in a 25 m radius around the actual ego, not
            # around the spot — for non-trivial spot_offset_xy the ego may be
            # far enough that the spot-centred ball would miss obstacles
            # immediately next to the start pose.
            ego_loc = ego.get_transform().location
            obstacles = extract_static_obstacles(
                world,
                ego_xy_world=(ego_loc.x, ego_loc.y),
                radius_m=25.0,
                frame=frame,
                ignore_ids=(ego.id,),
            )

            # Plan in the planner frame.
            t0 = time.perf_counter()
            waypoints, planner_metrics = plan_parking(parking_type, cc, pc, obstacles)
            planning_time = time.perf_counter() - t0
            if len(waypoints) < 2:
                return DemoResult(
                    planner_success=False,
                    planner_message=f"CARLA: planner returned no usable path ({planner_metrics})",
                    planning_time_s=planning_time,
                    planned_waypoints=[],
                    executed_poses=[],
                    executed_success=False,
                    executed_message="skipped (no path)",
                    final_pos_error_m=float("nan"),
                    final_heading_error_deg=float("nan"),
                    mean_cte_m=float("nan"),
                    max_cte_m=float("nan"),
                )

            ctl = CarlaPurePursuitController(waypoints, cc, ctl_cfg)
            dt = conn.fixed_delta_seconds
            poses: List[Tuple[float, float, float]] = []
            ctes: List[float] = []
            # Seed the executed-pose list with the planner-frame start pose so
            # downstream code can always read poses[-1] even if max_seconds is
            # 0 or conn.tick() throws on the very first iteration.
            poses.append(pose_from_actor(ego, frame))

            cmd: Optional[ControlCommand] = None
            elapsed = 0.0
            while elapsed < max_seconds:
                conn.tick()
                pose = pose_from_actor(ego, frame)
                v = ego.get_velocity()
                speed = math.sqrt(v.x ** 2 + v.y ** 2)
                cmd = ctl.step(pose, speed, dt)
                ego.apply_control(control_to_carla(cmd))
                poses.append(pose)
                ctes.append(_cte(waypoints, pose[0], pose[1]))
                if cmd.done:
                    break
                elapsed += dt

            if cmd is None:
                exec_msg = "carla: control loop never executed (max_seconds <= 0)"
            elif cmd.done:
                exec_msg = "carla: controller reported done"
            else:
                exec_msg = f"carla: aborted at {elapsed:.1f}s (no convergence)"

            goal = waypoints[-1]
            fx, fy, fth = poses[-1]
            pos_err = math.hypot(fx - goal.x, fy - goal.y)
            head_err = math.degrees(abs(_angle_diff(fth, goal.theta)))
            mean_cte = sum(ctes) / len(ctes) if ctes else 0.0
            max_cte = max(ctes) if ctes else 0.0

            return DemoResult(
                planner_success=True,
                planner_message="planner OK",
                planning_time_s=planning_time,
                planned_waypoints=list(waypoints),
                executed_poses=poses,
                executed_success=bool(cmd and cmd.done and pos_err < 0.5),
                executed_message=exec_msg,
                final_pos_error_m=pos_err,
                final_heading_error_deg=head_err,
                mean_cte_m=mean_cte,
                max_cte_m=max_cte,
            )
        finally:
            try:
                ego.destroy()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("perpendicular", "parallel"),
                        default="perpendicular")
    parser.add_argument("--scenario",
                        choices=("none", "entry_blocker", "tight_lane",
                                 "pillar_near_entry", "parked_cars"),
                        default="none",
                        help="Only meaningful in --dry-run; in --carla mode "
                             "obstacles come from the live world.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--carla", action="store_true",
                       help="Run on a live CARLA server.")
    group.add_argument("--dry-run", action="store_true", default=True,
                       help="Run the pipeline against our internal bicycle "
                            "simulator (no CARLA required).")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--probe", action="store_true",
                        help="Just probe whether the CARLA PythonAPI is importable.")
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="Max wall-clock seconds for the control loop "
                             "(--carla mode only).")
    parser.add_argument("--ctl-forward-speed", type=float, default=None,
                        help="Override ControlConfig.target_forward_speed (m/s).")
    parser.add_argument("--ctl-reverse-speed", type=float, default=None,
                        help="Override ControlConfig.target_reverse_speed (m/s).")
    parser.add_argument("--ctl-throttle-kp", type=float, default=None,
                        help="Override ControlConfig.throttle_kp.")
    parser.add_argument("--ctl-brake-kp", type=float, default=None,
                        help="Override ControlConfig.brake_kp.")
    parser.add_argument("--ctl-lookahead", type=float, default=None,
                        help="Override ControlConfig.lookahead_base (m).")
    return parser.parse_args(argv)


def _ctl_cfg_from_args(args: argparse.Namespace) -> Optional[ControlConfig]:
    overrides = {
        "target_forward_speed": args.ctl_forward_speed,
        "target_reverse_speed": args.ctl_reverse_speed,
        "throttle_kp": args.ctl_throttle_kp,
        "brake_kp": args.ctl_brake_kp,
        "lookahead_base": args.ctl_lookahead,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if not overrides:
        return None
    return ControlConfig(**overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.probe:
        print(f"CARLA_AVAILABLE={CARLA_AVAILABLE}")
        return 0

    ctl_cfg = _ctl_cfg_from_args(args)

    if args.carla:
        if not CARLA_AVAILABLE:
            print("carla_demo: CARLA PythonAPI not installed. Use --dry-run "
                  "or `pip install carla` matching your CARLA version.",
                  file=sys.stderr)
            return 2
        result = run_carla(
            host=args.host,
            port=args.port,
            town=args.town,
            parking_type=args.mode,
            max_seconds=args.max_seconds,
            ctl_cfg=ctl_cfg,
        )
    else:
        result = run_dry(
            parking_type=args.mode,
            obstacle_scenario=args.scenario,
            ctl_cfg=ctl_cfg,
        )

    print("carla_demo:", result.summary())
    return 0 if result.executed_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
