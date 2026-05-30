"""
Static parking-lot scenarios used by the planner and visualizers.

The rectangles represent occupied space in world coordinates. They can model
parked vehicles, pillars, carts, or blocked cells from a local occupancy grid.
"""
from __future__ import annotations

from typing import List

from parking_lot import ParkingLot, Rect


SCENARIO_NAMES = (
    "none",
    "entry_blocker",
    "tight_lane",
    "pillar_near_entry",
    "parked_cars",
)


SCENARIO_ALIASES = {
    "clear": "none",
    "obstacle": "entry_blocker",
    "tight": "tight_lane",
}


def normalize_scenario(name: str) -> str:
    scenario = SCENARIO_ALIASES.get(name, name)
    if scenario not in SCENARIO_NAMES:
        valid = ", ".join((*SCENARIO_ALIASES.keys(), *SCENARIO_NAMES))
        raise ValueError(f"Unknown scenario {name!r}. Valid values: {valid}")
    return scenario


def obstacles_for(lot: ParkingLot) -> List[Rect]:
    scenario = normalize_scenario(lot.pc.obstacle_scenario)
    if scenario == "entry_blocker":
        return _entry_blocker(lot)
    if scenario == "pillar_near_entry":
        return _pillar_near_entry(lot)
    if scenario == "parked_cars":
        return _parked_cars(lot)
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


def _pillar_near_entry(lot: ParkingLot) -> List[Rect]:
    lane = lot.lane_rect
    spot = lot.spot_rect
    size = 0.45

    if lot.pc.parking_type == "perpendicular":
        x = min(spot.right + 0.65, lane.right - size - 0.5)
        y = max(lane.y + 0.35, lane.top - 1.15)
        return [Rect(x, y, size, size)]

    x = max(lane.x + 0.5, spot.x - 0.85)
    y = lane.y + 0.35
    return [Rect(x, y, size, size)]


def _parked_cars(lot: ParkingLot) -> List[Rect]:
    lane = lot.lane_rect
    spot = lot.spot_rect

    if lot.pc.parking_type == "perpendicular":
        h = min(0.75, lane.h * 0.22)
        w = 1.65
        y = max(lane.y + 0.25, lane.top - h - 0.25)
        left_x = max(lane.x + 0.4, spot.x - w - 0.55)
        right_x = min(lane.right - w - 0.4, spot.right + 0.55)
        return [Rect(left_x, y, w, h), Rect(right_x, y, w, h)]

    h = min(1.25, lane.h * 0.35)
    w = 0.85
    y = lane.y + 0.15
    left_x = max(lane.x + 0.3, spot.x - w - 0.55)
    right_x = min(lane.right - w - 0.3, spot.right + 0.55)
    return [Rect(left_x, y, w, h), Rect(right_x, y, w, h)]
