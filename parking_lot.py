"""
World-space layout geometry for the parking scene.
Coordinate system: +x right, +y up (math convention).
All units in meters.
"""
import math
from dataclasses import dataclass
from typing import Tuple

from config import CarConfig, ParkingConfig


@dataclass
class Rect:
    x: float      # left edge
    y: float      # bottom edge
    w: float
    h: float

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h


class ParkingLot:
    """
    Computes world-space geometry for a given configuration.

    Perpendicular layout (倒車入庫):
      - Lane: horizontal band at y=0..lane_width
      - Spot: rectangle above the lane, centred along a fixed-length lane
      - Car start: in the lane to the left of the spot, heading +x

    Parallel layout (路邊停車):
      - Lane: horizontal band at y=0..lane_width
      - Spot: below the lane, centred along a fixed-length lane
      - Car start: in the lane to the left of the spot, heading +x
    """

    # Extra clearance added around the scene for canvas/pygame margin
    SCENE_PADDING = 2.0

    # Fixed lane lengths based on maximum slider values so that
    # changing car_length or spot_width/length never resizes the road.
    # Values: max_car_len=5.0, max_spot_w=3.0, max_spot_len=6.0
    _PERP_LANE_LEN = 5.0 * 4 + 3.0          # 23.0 m  (approach + max_spot_w + exit)
    _PAR_LANE_LEN  = 5.0 * 2.5 + 6.0 + 5.0  # 23.5 m  (approach + max_spot_len + exit)

    def __init__(self, parking_config: ParkingConfig, car_config: CarConfig):
        self.pc = parking_config
        self.cc = car_config
        self._build()

    def _build(self):
        pc, cc = self.pc, self.cc

        if pc.parking_type == "perpendicular":
            self._build_perpendicular()
        else:
            self._build_parallel()

    # ------------------------------------------------------------------
    # Perpendicular (倒車入庫)
    # ------------------------------------------------------------------
    def _build_perpendicular(self):
        pc, cc = self.pc, self.cc

        lane_len = self._PERP_LANE_LEN  # fixed — never changes with sliders

        self.lane_rect = Rect(0.0, 0.0, lane_len, pc.lane_width)

        # Spot centred in the fixed-length lane
        spot_x = (lane_len - pc.spot_width) / 2
        self.spot_rect = Rect(spot_x, pc.lane_width, pc.spot_width, pc.spot_length)

        # Car starts to the left of the spot centre, parallel to road
        self.car_start_pose: Tuple[float, float, float] = (
            spot_x / 2,
            pc.lane_width / 2,
            0.0,
        )

        self.scene_w = lane_len + self.SCENE_PADDING
        self.scene_h = pc.lane_width + pc.spot_length + self.SCENE_PADDING

    # ------------------------------------------------------------------
    # Parallel (路邊停車)
    # ------------------------------------------------------------------
    def _build_parallel(self):
        pc, cc = self.pc, self.cc

        lane_len = self._PAR_LANE_LEN  # fixed — never changes with sliders

        self.lane_rect = Rect(0.0, 0.0, lane_len, pc.lane_width)

        # Spot centred along the fixed-length lane, below it
        spot_x = (lane_len - pc.spot_length) / 2
        self.spot_rect = Rect(spot_x, -pc.spot_width, pc.spot_length, pc.spot_width)

        # Car starts to the left of the spot, heading +x
        self.car_start_pose = (
            spot_x / 2,
            pc.lane_width / 2,
            0.0,
        )

        self.scene_w = lane_len + self.SCENE_PADDING
        self.scene_h = pc.lane_width + pc.spot_width + self.SCENE_PADDING

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def car_fits(self) -> bool:
        """True if the car can geometrically fit inside the spot."""
        pc, cc = self.pc, self.cc
        if pc.parking_type == "perpendicular":
            return cc.width <= pc.spot_width and cc.length <= pc.spot_length
        else:
            # For parallel: car length ≤ spot length, car width ≤ spot width
            return cc.length <= pc.spot_length and cc.width <= pc.spot_width

    def car_corners(self, pose: Tuple[float, float, float]) -> list:
        """Return 4 world-space corners of the car at the given pose (x, y, theta).
        Pose is the rear-axle center."""
        x, y, theta = pose
        cc = self.cc
        L, W = cc.length, cc.width

        # Rear-axle is offset from car center by half the rear overhang.
        # For simplicity treat rear-axle as rear edge center.
        # Four corners relative to rear-axle:
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def rotated(dx, dy):
            return (
                x + dx * cos_t - dy * sin_t,
                y + dx * sin_t + dy * cos_t,
            )

        half_w = W / 2
        return [
            rotated(0,  half_w),     # rear-left
            rotated(L,  half_w),     # front-left
            rotated(L, -half_w),     # front-right
            rotated(0, -half_w),     # rear-right
        ]
