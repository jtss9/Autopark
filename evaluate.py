"""
Batch evaluation for the parking planner.

This is intended for final-project reporting: run several parking scenarios and
print planner metrics as CSV so results can be copied into a table or plotted.
"""
from __future__ import annotations

import csv
import sys

from config import CarConfig, ParkingConfig
from trajectory import plan_trajectory


CASES = [
    ("perpendicular_clear", ParkingConfig(parking_type="perpendicular")),
    ("perpendicular_obstacle", ParkingConfig(
        parking_type="perpendicular",
        obstacle_scenario="entry_blocker",
    )),
    ("parallel_clear", ParkingConfig(parking_type="parallel")),
    ("parallel_obstacle", ParkingConfig(
        parking_type="parallel",
        obstacle_scenario="entry_blocker",
    )),
]


def main() -> int:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "case",
            "feasible",
            "message",
            "planning_time_s",
            "iterations",
            "expanded_states",
            "path_length_m",
            "waypoints",
            "final_pos_error_m",
            "final_heading_error_deg",
            "fully_in_spot",
            "obstacles",
        ],
    )
    writer.writeheader()

    car = CarConfig()
    for name, parking in CASES:
        result = plan_trajectory(parking, car, planner="hybrid_astar")
        row = {
            "case": name,
            "feasible": result.feasible,
            "message": result.message,
        }
        row.update(result.metrics)
        writer.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

