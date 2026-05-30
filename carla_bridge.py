"""
CARLA simulator bridge for the autonomous parking pipeline.

This module is the thin adapter between our planner/controller stack and a
running CARLA server. It is deliberately split so that the rest of the project
can be imported and tested without `carla` installed:

  - All `carla` imports are lazy (inside functions / methods), so importing
    this module on a machine without the CARLA PythonAPI works fine and just
    sets `CARLA_AVAILABLE = False`.
  - The bridge translates between CARLA world coordinates (left-handed) and
    our planner's right-handed math coordinates (+x right, +y up, units in m).
  - Obstacles are extracted from `world.get_actors()` (vehicles, walls,
    static props) within a configurable radius of the ego vehicle. A
    LiDAR-based path is also available for sensor-realistic experiments.

Public API:
    CARLA_AVAILABLE              - bool, True when `import carla` succeeded.
    require_carla()              - raises a clear error if CARLA is missing.
    CarlaConnection              - context manager wrapping the client/world.
    LocalFrame                   - planning-frame transform (CARLA <-> planner).
    spawn_ego(world, transform)  - blueprint lookup + spawn helper.
    pose_from_actor(actor, frame)- read (x, y, theta) in the planner frame.
    extract_static_obstacles(...)- world actors -> Rect[] for planner.
    occupancy_from_lidar(...)    - LiDAR cloud -> Rect[] cells.

This file deliberately has zero behavioural dependencies on the rest of the
project (no `trajectory`, `hybrid_astar`, etc.) so it can be reused as a
standalone library.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from parking_lot import Rect


# ---------------------------------------------------------------------------
# Lazy import probe
# ---------------------------------------------------------------------------
try:
    import carla as _carla  # noqa: F401
    CARLA_AVAILABLE = True
except Exception:  # ImportError, but also CARLA's odd egg/wheel failures
    CARLA_AVAILABLE = False


def require_carla() -> None:
    """Raise a clear error if CARLA is not importable."""
    if not CARLA_AVAILABLE:
        raise RuntimeError(
            "CARLA PythonAPI not available. Install the matching wheel for "
            "your CARLA version (typically `pip install carla==0.9.15` on "
            "Python 3.8-3.10), and ensure a CARLA server is running."
        )


# ---------------------------------------------------------------------------
# Planning frame
# ---------------------------------------------------------------------------
@dataclass
class LocalFrame:
    """
    Transform between CARLA world coordinates and our planner's local frame.

    CARLA's coordinate system is left-handed: +X forward, +Y right, +Z up
    (yaw in degrees, clockwise from +X). Our planner uses standard math
    convention: +x right, +y up, theta in radians counterclockwise from +x.

    The bridge picks an `origin` point in CARLA world coordinates and a
    `yaw_offset_deg` that defines what "forward" means for the parking lot
    so that the planner sees a rectangular lane aligned with +x.

    Usage:
        frame = LocalFrame(origin_xy=(spot.x, spot.y), yaw_offset_deg=spot_yaw)
        local_pose = frame.world_to_local(actor.get_transform())
    """
    origin_xy: Tuple[float, float]
    yaw_offset_deg: float = 0.0

    def world_to_local(self, transform) -> Tuple[float, float, float]:
        """Convert a `carla.Transform` to (x, y, theta) in planner frame."""
        loc = transform.location
        # Translate to origin, then rotate by -yaw_offset to align lane with +x.
        dx = loc.x - self.origin_xy[0]
        dy = loc.y - self.origin_xy[1]
        # CARLA's Y axis is flipped relative to math convention; negate to
        # convert handedness so that our planner sees a standard right-handed
        # frame with +y up.
        dy = -dy
        yaw_rad = math.radians(-(transform.rotation.yaw + self.yaw_offset_deg))
        cos_t = math.cos(math.radians(-self.yaw_offset_deg))
        sin_t = math.sin(math.radians(-self.yaw_offset_deg))
        x_local = cos_t * dx - sin_t * dy
        y_local = sin_t * dx + cos_t * dy
        return x_local, y_local, yaw_rad

    def local_to_world_xy(self, x: float, y: float) -> Tuple[float, float]:
        """Convert a planner-frame (x, y) point back to CARLA world (x, y)."""
        cos_t = math.cos(math.radians(self.yaw_offset_deg))
        sin_t = math.sin(math.radians(self.yaw_offset_deg))
        dx = cos_t * x - sin_t * y
        dy = sin_t * x + cos_t * y
        wx = self.origin_xy[0] + dx
        wy = self.origin_xy[1] - dy  # undo handedness flip
        return wx, wy

    def local_theta_to_world_yaw_deg(self, theta_rad: float) -> float:
        return -(math.degrees(theta_rad)) - self.yaw_offset_deg


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------
class CarlaConnection:
    """
    Context manager that owns a `carla.Client` and (optionally) loads a map.

    The client and world handles are accessible via `.client` and `.world`
    after entering the context. Exiting restores the world to async mode
    so the CARLA server is not left in synchronous mode if our process dies.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout_s: float = 10.0,
        town: Optional[str] = None,
        synchronous: bool = True,
        fixed_delta_seconds: float = 0.05,
    ):
        require_carla()
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.town = town
        self.synchronous = synchronous
        self.fixed_delta_seconds = fixed_delta_seconds
        self.client = None
        self.world = None
        self._prev_settings = None

    def __enter__(self):
        import carla
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout_s)
        if self.town:
            self.world = self.client.load_world(self.town)
        else:
            self.world = self.client.get_world()
        if self.synchronous:
            settings = self.world.get_settings()
            self._prev_settings = settings
            new_settings = self.world.get_settings()
            new_settings.synchronous_mode = True
            new_settings.fixed_delta_seconds = self.fixed_delta_seconds
            self.world.apply_settings(new_settings)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.world is not None and self._prev_settings is not None:
                self.world.apply_settings(self._prev_settings)
        except Exception:
            pass
        return False

    def tick(self) -> None:
        """Advance the simulation one step in synchronous mode."""
        if self.world is None:
            return
        if self.synchronous:
            self.world.tick()
        else:
            self.world.wait_for_tick()


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------
def spawn_ego(
    world,
    transform,
    blueprint_filter: str = "vehicle.tesla.model3",
):
    """Spawn an ego vehicle at the given `carla.Transform`."""
    require_carla()
    blueprint_library = world.get_blueprint_library()
    candidates = blueprint_library.filter(blueprint_filter)
    if not candidates:
        raise RuntimeError(
            f"No vehicle blueprints match {blueprint_filter!r}; try "
            "'vehicle.*' or a specific model id."
        )
    bp = candidates[0]
    bp.set_attribute("role_name", "ego")
    actor = world.try_spawn_actor(bp, transform)
    if actor is None:
        raise RuntimeError(
            f"Could not spawn vehicle at {transform.location}. The spot may "
            "be occupied by another actor or outside the map."
        )
    return actor


def pose_from_actor(actor, frame: LocalFrame) -> Tuple[float, float, float]:
    """Return (x, y, theta) for an actor in the planner frame."""
    return frame.world_to_local(actor.get_transform())


# ---------------------------------------------------------------------------
# Obstacle extraction
# ---------------------------------------------------------------------------
def extract_static_obstacles(
    world,
    ego_xy_world: Tuple[float, float],
    radius_m: float,
    frame: LocalFrame,
    actor_filters: Sequence[str] = ("vehicle.*", "static.prop.*"),
    ignore_ids: Iterable[int] = (),
) -> List[Rect]:
    """
    Walk `world.get_actors()` and convert nearby bounding boxes into Rect[]
    in the planner frame. Skip actors whose id is in `ignore_ids` (typically
    the ego vehicle itself).
    """
    require_carla()
    ignore_set = set(ignore_ids)
    rects: List[Rect] = []
    actors = world.get_actors()
    for fil in actor_filters:
        for actor in actors.filter(fil):
            if actor.id in ignore_set:
                continue
            t = actor.get_transform()
            loc = t.location
            dx = loc.x - ego_xy_world[0]
            dy = loc.y - ego_xy_world[1]
            if math.hypot(dx, dy) > radius_m:
                continue
            bbox = getattr(actor, "bounding_box", None)
            if bbox is None:
                continue
            # Use the AABB in world XY (yaw-rotated extents). We over-approximate
            # by taking the larger of (extent.x, extent.y) so a yaw-rotated bbox
            # is conservatively covered. This is fine for parking-lot obstacles
            # where exact shape is not critical and the planner's car-corner
            # collision check eats any small over-conservatism.
            ext_half = max(bbox.extent.x, bbox.extent.y)
            x_local, y_local, _ = frame.world_to_local(t)
            rects.append(
                Rect(
                    x_local - ext_half,
                    y_local - ext_half,
                    2 * ext_half,
                    2 * ext_half,
                )
            )
    return rects


def occupancy_from_lidar(
    points: Sequence[Tuple[float, float, float]],
    frame: LocalFrame,
    ego_xy_world: Tuple[float, float],
    cell_size: float = 0.5,
    z_min: float = 0.2,
    z_max: float = 2.5,
    radius_m: float = 25.0,
) -> List[Rect]:
    """
    Convert a frame of LiDAR points (world coordinates) into Rect[] obstacle
    cells in the planner frame.

    `points` is a sequence of (x, y, z) tuples already projected into CARLA
    world coordinates. We filter by:
      - height band [z_min, z_max] (drop ground returns and overhanging signs)
      - planar distance from ego (drop far returns)
    Remaining points are voxelised to `cell_size` square cells; each occupied
    cell becomes a single small Rect.
    """
    occupied: set = set()
    half = cell_size / 2
    for px, py, pz in points:
        if pz < z_min or pz > z_max:
            continue
        if math.hypot(px - ego_xy_world[0], py - ego_xy_world[1]) > radius_m:
            continue
        # Use the lidar point as a CARLA Location-like; we only need x, y.
        # Build a tiny transform-like wrapper for frame.world_to_local.
        class _T:
            class location:
                x = px
                y = py
                z = 0.0
            class rotation:
                yaw = 0.0
        local_x, local_y, _ = frame.world_to_local(_T())
        ix = int(round(local_x / cell_size))
        iy = int(round(local_y / cell_size))
        occupied.add((ix, iy))

    rects: List[Rect] = []
    for ix, iy in occupied:
        cx = ix * cell_size
        cy = iy * cell_size
        rects.append(Rect(cx - half, cy - half, cell_size, cell_size))
    return rects


# ---------------------------------------------------------------------------
# Vehicle parameters
# ---------------------------------------------------------------------------
def car_config_from_vehicle(vehicle) -> Tuple[float, float, float]:
    """
    Return (length, width, wheelbase) of a CARLA vehicle actor.
    """
    require_carla()
    bbox = vehicle.bounding_box
    length = 2.0 * bbox.extent.x
    width = 2.0 * bbox.extent.y
    # CARLA does not always expose wheelbase directly; fall back to the same
    # heuristic our CarConfig uses.
    wheelbase = length * 0.65
    return length, width, wheelbase
