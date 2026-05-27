"""
Static parking-lot scenarios used by the planner and visualizers.

The rectangles represent occupied space in world coordinates. They can model
parked vehicles, pillars, carts, or blocked cells from a local occupancy grid.
"""
from __future__ import annotations

from typing import List

from parking_lot import ParkingLot, Rect


def obstacles_for(lot: ParkingLot) -> List[Rect]:
    scenario = lot.pc.obstacle_scenario
    if scenario == "entry_blocker":
        return _entry_blocker(lot)
    return []


def _entry_blocker(lot: ParkingLot) -> List[Rect]:
    lane = lot.lane_rect
    spot = lot.spot_rect

    if lot.pc.parking_type == "perpendicular":
        # A small occupied patch near the right-side approach. The planner must
        # leave enough clearance while still entering the selected spot.
        w = 1.0
        h = min(1.0, lane.h * 0.28)
        x = spot.right + 1.2
        y = lane.y + 0.25
        return [Rect(x, y, w, h)]

    # Parallel: obstacle behind the selected curb-side spot, similar to a parked
    # vehicle limiting the rear approach space.
    w = min(1.0, spot.w * 0.18)
    h = spot.h
    x = max(lane.x + 0.2, spot.x - w - 0.35)
    y = spot.y
    return [Rect(x, y, w, h)]

