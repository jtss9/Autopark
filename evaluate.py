"""
Batch evaluation for the parking planners.

The evaluator is intentionally GUI-free so it can produce CSV tables for the
final report and for repeatable command-line smoke tests.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Sequence

from config import CarConfig, ParkingConfig
from geom import angle_diff as _angle_diff, path_length as _path_length_seq
from parking_lot import ParkingLot
from scenarios import SCENARIO_NAMES, normalize_scenario
from trajectory import TrajectoryResult, plan_trajectory


FIELDNAMES = [
    "mode",
    "scenario",
    "planner",
    "sweep",
    "lane_width",
    "spot_length",
    "spot_width",
    "car_length",
    "car_width",
    "feasible",
    "success",
    "message",
    "planning_time_s",
    "iterations",
    "expanded_states",
    "path_length_m",
    "raw_path_length_m",
    "smoothed_path_length_m",
    "waypoints",
    "raw_waypoints",
    "smoothed_waypoints",
    "final_pos_error_m",
    "final_heading_error_deg",
    "fully_in_spot",
    "obstacles",
    "rs_shot_attempts",
    "rs_shot_successes",
    "used_analytic_shot",
    "tracker_success",
    "mean_cte_m",
    "max_cte_m",
    "exec_final_pos_error_m",
    "exec_final_heading_error_deg",
    "exec_fully_in_spot",
    "cusps",
    "training_time_s",
    "successful_episodes",
]

MODE_CHOICES = ("perpendicular", "parallel", "all")
SCENARIO_CHOICES = (
    "clear",
    "obstacle",
    "tight",
    "all",
    *SCENARIO_NAMES,
)
SWEEP_CHOICES = ("lane_width", "spot_size", "car_size", "none")
PLANNER_CHOICES = ("baseline", "hybrid_astar", "qlearn", "all")


def _path_length(result: TrajectoryResult) -> float:
    return _path_length_seq(result.waypoints)


def _goal_pose(lot: ParkingLot) -> tuple[float, float, float]:
    spot = lot.spot_rect
    if lot.pc.parking_type == "parallel":
        return spot.x + 0.15, spot.y + spot.h / 2, 0.0
    return spot.x + spot.w / 2, spot.top - 0.15, -math.pi / 2


def _fully_in_spot(lot: ParkingLot, result: TrajectoryResult) -> bool:
    if not result.waypoints:
        return False
    spot = lot.spot_rect
    final = result.waypoints[-1]
    return all(
        spot.x <= x <= spot.right and spot.y <= y <= spot.top
        for x, y in lot.car_corners((final.x, final.y, final.theta))
    )


def _metric(result: TrajectoryResult, name: str, default=""):
    return result.metrics.get(name, default)


def _base_row(
    pc: ParkingConfig,
    cc: CarConfig,
    planner: str,
    sweep: str,
) -> dict:
    return {
        "mode": pc.parking_type,
        "scenario": normalize_scenario(pc.obstacle_scenario),
        "planner": planner,
        "sweep": sweep,
        "lane_width": pc.lane_width,
        "spot_length": pc.spot_length,
        "spot_width": pc.spot_width,
        "car_length": cc.length,
        "car_width": cc.width,
    }


def _result_row(
    pc: ParkingConfig,
    cc: CarConfig,
    planner: str,
    sweep: str,
    result: TrajectoryResult,
    elapsed_s: float,
) -> dict:
    lot = ParkingLot(pc, cc)
    goal = _goal_pose(lot)
    full_spot = bool(_metric(result, "fully_in_spot", _fully_in_spot(lot, result)))
    final_pos_error = ""
    final_heading_error = ""
    if result.waypoints:
        final = result.waypoints[-1]
        final_pos_error = math.hypot(final.x - goal[0], final.y - goal[1])
        final_heading_error = math.degrees(abs(_angle_diff(final.theta, goal[2])))

    success = bool(result.feasible and full_spot)
    message = result.message
    if result.feasible and not full_spot:
        message = "Final car is not fully inside the parking spot."

    row = _base_row(pc, cc, planner, sweep)
    row.update({
        "feasible": result.feasible,
        "success": success,
        "message": message,
        "planning_time_s": _metric(result, "planning_time_s", elapsed_s),
        "iterations": _metric(result, "iterations"),
        "expanded_states": _metric(result, "expanded_states"),
        "path_length_m": _metric(result, "path_length_m", _path_length(result)),
        "raw_path_length_m": _metric(result, "raw_path_length_m"),
        "smoothed_path_length_m": _metric(result, "smoothed_path_length_m"),
        "waypoints": _metric(result, "waypoints", len(result.waypoints)),
        "raw_waypoints": _metric(result, "raw_waypoints"),
        "smoothed_waypoints": _metric(result, "smoothed_waypoints"),
        "final_pos_error_m": _metric(result, "final_pos_error_m", final_pos_error),
        "final_heading_error_deg": _metric(
            result,
            "final_heading_error_deg",
            final_heading_error,
        ),
        "fully_in_spot": full_spot,
        "obstacles": _metric(result, "obstacles", ""),
        "rs_shot_attempts": _metric(result, "rs_shot_attempts"),
        "rs_shot_successes": _metric(result, "rs_shot_successes"),
        "used_analytic_shot": _metric(result, "used_analytic_shot"),
        "training_time_s": _metric(result, "training_time_s"),
        "successful_episodes": _metric(result, "successful_episodes"),
    })
    tm = result.tracking_metrics or {}
    if tm:
        row.update({
            "tracker_success": tm.get("tracker_success"),
            "mean_cte_m": tm.get("mean_cte_m"),
            "max_cte_m": tm.get("max_cte_m"),
            "exec_final_pos_error_m": tm.get("exec_final_pos_error_m"),
            "exec_final_heading_error_deg": tm.get("exec_final_heading_error_deg"),
            "exec_fully_in_spot": tm.get("exec_fully_in_spot"),
            "cusps": tm.get("cusps"),
        })
    return row


def _skip_row(
    pc: ParkingConfig,
    cc: CarConfig,
    planner: str,
    sweep: str,
    reason: str,
) -> dict:
    row = _base_row(pc, cc, planner, sweep)
    row.update({
        "feasible": False,
        "success": False,
        "message": reason,
    })
    return row


def _modes(selected: str) -> Sequence[str]:
    if selected == "all":
        return ("perpendicular", "parallel")
    return (selected,)


def _scenarios(selected: str) -> Sequence[str]:
    if selected == "all":
        return SCENARIO_NAMES
    return (normalize_scenario(selected),)


def _planners(selected: str) -> Sequence[str]:
    if selected == "all":
        return ("baseline", "hybrid_astar", "qlearn")
    return (selected,)


def _sweep_configs(
    base_pc: ParkingConfig,
    base_cc: CarConfig,
    sweep: str,
) -> Iterable[tuple[ParkingConfig, CarConfig]]:
    if sweep == "lane_width":
        for lane_width in (3.5, 4.0, 4.5, 5.0, 5.5):
            yield replace(base_pc, lane_width=lane_width), base_cc
        return

    if sweep == "spot_size":
        for spot_width in (2.0, 2.3, 2.5, 2.8, 3.0):
            for spot_length in (5.0, 5.5, 6.0):
                yield replace(
                    base_pc,
                    spot_width=spot_width,
                    spot_length=spot_length,
                ), base_cc
        return

    if sweep == "car_size":
        for car_length in (3.5, 4.0, 4.5, 5.0):
            yield base_pc, replace(base_cc, length=car_length)
        return

    yield base_pc, base_cc


def _scenario_config(mode: str, scenario: str) -> ParkingConfig:
    pc = ParkingConfig(parking_type=mode, obstacle_scenario=scenario)
    if scenario == "tight_lane":
        pc = replace(pc, lane_width=3.8)
    return pc


def _can_run_baseline(pc: ParkingConfig) -> tuple[bool, str]:
    if pc.parking_type != "perpendicular":
        return False, "Skipped: baseline only supports perpendicular parking."
    if normalize_scenario(pc.obstacle_scenario) != "none":
        return False, "Skipped: baseline comparison is only defined for clear cases."
    if importlib.util.find_spec("scipy") is None:
        return False, "Skipped: SciPy is not installed, baseline MPC unavailable."
    return True, ""


def run_case(
    pc: ParkingConfig,
    cc: CarConfig,
    planner: str,
    sweep: str,
    track: bool = False,
) -> dict:
    if planner == "baseline":
        can_run, reason = _can_run_baseline(pc)
        if not can_run:
            return _skip_row(pc, cc, planner, sweep, reason)
        # "baseline" in the evaluator means the MPC family; pick single/multi
        # by lane width and pass it as the explicit planner so plan_trajectory
        # routes to the MPC branch (not the new pc.planner fallback).
        requested_planner = "multi" if pc.lane_width < 4.5 else "single"
    elif planner == "qlearn":
        requested_planner = "qlearn"
    else:
        requested_planner = "hybrid_astar"

    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = plan_trajectory(
                pc, cc, planner=requested_planner, track=track,
            )
    except Exception as exc:  # keep batch evaluation resilient for reports
        return _skip_row(pc, cc, planner, sweep, f"Planner crashed: {exc}")
    elapsed = time.perf_counter() - started
    return _result_row(pc, cc, planner, sweep, result, elapsed)


def build_rows(args: argparse.Namespace) -> List[dict]:
    rows: List[dict] = []
    base_cc = CarConfig()
    for mode in _modes(args.mode):
        for scenario in _scenarios(args.scenario):
            base_pc = _scenario_config(mode, scenario)
            for pc, cc in _sweep_configs(base_pc, base_cc, args.sweep):
                for planner in _planners(args.planner):
                    rows.append(run_case(pc, cc, planner, args.sweep, track=args.track))
    return rows


def write_rows(rows: Sequence[dict], output: str | None) -> None:
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def print_summary(rows: Sequence[dict], stream) -> None:
    total = len(rows)
    successes = [r for r in rows if r.get("success") is True]

    def _numeric_col(rows: Sequence[dict], col: str) -> list:
        # Use a `is not None` / `!= ""` filter rather than truthiness so that
        # legitimate zero values (planning_time_s=0.0 from a cached run,
        # final_pos_error_m=0.0 from an exact RS shot) are included in the
        # average instead of being silently dropped.
        out = []
        for r in rows:
            v = r.get(col)
            if v is None or v == "":
                continue
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out

    numeric_time = _numeric_col(rows, "planning_time_s")
    numeric_path = _numeric_col(rows, "path_length_m")
    numeric_error = _numeric_col(rows, "final_pos_error_m")
    full_spot = [r for r in rows if r.get("fully_in_spot") is True]

    def avg(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    print(
        "summary,"
        f"rows={total},"
        f"success_rate={len(successes) / total if total else 0.0:.3f},"
        f"avg_planning_time_s={avg(numeric_time):.3f},"
        f"avg_path_length_m={avg(numeric_path):.3f},"
        f"avg_final_pos_error_m={avg(numeric_error):.3f},"
        f"full_spot_success_rate={len(full_spot) / total if total else 0.0:.3f}",
        file=stream,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODE_CHOICES, default="all")
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="clear")
    parser.add_argument("--sweep", choices=SWEEP_CHOICES, default="none")
    parser.add_argument("--planner", choices=PLANNER_CHOICES, default="all")
    parser.add_argument("--output", help="Write CSV rows to this path instead of stdout.")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Also run the Pure Pursuit closed-loop tracker and record tracking metrics.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    rows = build_rows(args)
    write_rows(rows, args.output)
    print_summary(rows, sys.stdout if args.output else sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
