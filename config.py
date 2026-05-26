from dataclasses import dataclass, field
import math


@dataclass
class CarConfig:
    length: float = 4.5
    width: float = 1.8
    wheelbase: float = field(init=False)
    max_steer: float = field(init=False)

    def __post_init__(self):
        self.wheelbase = self.length * 0.65
        self.max_steer = math.radians(35)

    @property
    def min_turn_radius(self) -> float:
        return self.wheelbase / math.tan(self.max_steer)


@dataclass
class ParkingConfig:
    lane_width: float = 6.0
    spot_length: float = 6.0
    spot_width: float = 2.5
    parking_type: str = "perpendicular"  # "perpendicular" | "parallel"
    planner: str = "single"             # "single" | "multi"
