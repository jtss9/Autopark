"""
Bicycle kinematic model (rear-axle reference) and MPC path-following controller.

State   : (x, y, theta)  — rear-axle centre, heading in radians
Controls: v (m/s, negative = reverse)  |  delta (steering angle, rad)
Kinematics:
    x'     = v * cos(theta)
    y'     = v * sin(theta)
    theta' = v * tan(delta) / wheelbase
"""
import math
import numpy as np
from scipy.optimize import minimize

from config import CarConfig
from parking_lot import ParkingLot


class CarDynamics:
    def __init__(self, x: float, y: float, theta: float, cc: CarConfig):
        self.x     = float(x)
        self.y     = float(y)
        self.theta = float(theta)
        self.cc    = cc

    def step(self, v: float, delta: float, dt: float) -> None:
        delta = float(np.clip(delta, -self.cc.max_steer, self.cc.max_steer))
        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta += v * math.tan(delta) / self.cc.wheelbase * dt

    @property
    def pose(self):
        return (self.x, self.y, self.theta)


class MPCController:
    """
    Receding-horizon MPC that tracks a geometric reference path while
    penalising car-body boundary violations.

    Speed v is supplied by the caller so gear changes (forward / reverse)
    are handled externally.
    """

    def __init__(self, lot: ParkingLot, cc: CarConfig, ref_waypoints):
        """
        ref_waypoints : list of objects with .x  .y  .theta attributes
        """
        self.lot = lot
        self.cc  = cc
        self.ref = ref_waypoints
        self.ref_idx = 0

        self.N  = 5      # prediction horizon (steps)
        self.dt = 0.05   # seconds per MPC step

        # ── cost weights ──────────────────────────────────────────────────────
        self.w_pos      = 3.0    # position tracking
        self.w_heading  = 1.5    # heading tracking
        self.w_delta    = 0.02   # steering effort
        self.w_ddelta   = 0.30   # steering smoothness (rate penalty)
        self.w_boundary = 2000.0 # boundary violation

    # ── internal helpers ──────────────────────────────────────────────────────

    def _step(self, x, y, theta, v, delta):
        delta = float(np.clip(delta, -self.cc.max_steer, self.cc.max_steer))
        return (
            x + v * math.cos(theta) * self.dt,
            y + v * math.sin(theta) * self.dt,
            theta + v * math.tan(delta) / self.cc.wheelbase * self.dt,
        )

    def _boundary_penalty(self, x, y, theta) -> float:
        """
        Squared-distance penalty for every car corner that violates:
          - bottom of lane (y < 0)
          - top of spot   (y > spot.top)
          - left/right road edges
          - spot x-range when the corner is above the lane
        """
        corners = self.lot.car_corners((x, y, theta))
        lane = self.lot.lane_rect
        spot = self.lot.spot_rect
        pen = 0.0
        for cx, cy in corners:
            if cy < lane.y:
                pen += (lane.y - cy) ** 2
            if cy > spot.top:
                pen += (cy - spot.top) ** 2
            if cx < lane.x:
                pen += (lane.x - cx) ** 2
            if cx > lane.right:
                pen += (cx - lane.right) ** 2
            # above lane → must be within spot x-range
            if cy > lane.top:
                if cx < spot.x:
                    pen += (spot.x - cx) ** 2
                elif cx > spot.right:
                    pen += (cx - spot.right) ** 2
        return pen

    def _advance_ref_idx(self, x, y) -> None:
        """Move ref_idx forward to the closest reference point."""
        end = min(len(self.ref), self.ref_idx + 100)
        best, best_d = self.ref_idx, float('inf')
        for i in range(self.ref_idx, end):
            d = math.hypot(self.ref[i].x - x, self.ref[i].y - y)
            if d < best_d:
                best_d, best = d, i
        self.ref_idx = best

    def _lookahead_target(self):
        la = min(self.ref_idx + 8, len(self.ref) - 1)
        return self.ref[la]

    # ── public API ────────────────────────────────────────────────────────────

    def optimize(self, car: CarDynamics, v: float,
                 prev_delta: float = 0.0) -> float:
        """Return the optimal steering angle for the current step."""
        self._advance_ref_idx(car.x, car.y)
        tgt = self._lookahead_target()

        def cost(deltas):
            cx, cy, cth = car.x, car.y, car.theta
            pd = prev_delta
            total = 0.0
            for d in deltas:
                cx, cy, cth = self._step(cx, cy, cth, v, d)
                total += self.w_pos * ((cx - tgt.x) ** 2 + (cy - tgt.y) ** 2)
                dh = (cth - tgt.theta + math.pi) % (2 * math.pi) - math.pi
                total += self.w_heading * dh ** 2
                total += self.w_delta   * d ** 2
                total += self.w_ddelta  * (d - pd) ** 2
                total += self.w_boundary * self._boundary_penalty(cx, cy, cth)
                pd = d
            return total

        bounds = [(-self.cc.max_steer, self.cc.max_steer)] * self.N
        x0 = np.full(self.N, prev_delta * 0.5)   # warm start
        res = minimize(cost, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 40, 'ftol': 1e-5})
        return float(np.clip(res.x[0], -self.cc.max_steer, self.cc.max_steer))

    def corners_in_bounds(self, x: float, y: float, theta: float,
                          margin: float = 0.05) -> bool:
        """True if every car corner is within lane ∪ spot (with margin)."""
        corners = self.lot.car_corners((x, y, theta))
        lane = self.lot.lane_rect
        spot = self.lot.spot_rect
        for cx, cy in corners:
            if cy < lane.y - margin or cy > spot.top + margin:
                return False
            if cx < lane.x - margin or cx > lane.right + margin:
                return False
            if cy > lane.top + margin:
                if cx < spot.x - margin or cx > spot.right + margin:
                    return False
        return True
