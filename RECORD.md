# Work Record

## Branch

Created and switched to:

```text
mason-implement
```

## Project Scope Update

Created `PROJECT_SCOPE.md` to store the upgraded final-project direction:

- Reframed the project as GPS-denied autonomous parking in structured parking lots.
- Kept SLAM/map building as a supporting input rather than the main implementation burden.
- Defined the target stack: known/SLAM map, occupancy grid, parking goal selection, Hybrid A* planner, path smoothing, controller, and simulation/evaluation.
- Kept the original fixed geometric-arc MPC planner as the baseline.

## Algorithm Work

Added `hybrid_astar.py` with an initial map-based parking planner:

- Occupancy-grid abstraction backed by `ParkingLot`.
- Search over vehicle state `(x, y, theta)`.
- Forward and reverse bicycle-model motion primitives.
- Steering discretization.
- Full car-corner validity checks against lane/spot boundaries.
- `TrajectoryResult` / `Waypoint` output compatible with the existing simulator.
- Initial perpendicular-parking Hybrid A* path.

Updated `trajectory.py`:

- Added a `planner` argument to `plan_trajectory(...)`.
- Added opt-in dispatch for `planner="hybrid_astar"`.
- Left the existing MPC/geometric planner as the default baseline.

Updated `simulation.py`:

- Added `AUTOPARK_PLANNER` environment-variable support.
- The simulator can now run the new planner with:

```bash
AUTOPARK_PLANNER=hybrid_astar python main.py
```

- Added the selected planner name to the HUD.

## UI Work

Fixed the settings window dark-mode contrast issue in `settings_window.py`:

- Replaced mixed OS-default colors with explicit dark UI colors.
- Added readable foreground colors for labels, slider values, and radio buttons.
- Added darker panel and canvas colors.
- Replaced the harsh white parking-spot outline with a cyan outline.
- Improved contrast for preview annotations and car outlines.

## Documentation

Updated `README.md`:

- Mentioned the upgraded Hybrid A* project direction.
- Added the `AUTOPARK_PLANNER=hybrid_astar` run command.
- Added `hybrid_astar.py` and `PROJECT_SCOPE.md` to the file structure.
- Documented the new planner at a high level.

## Validation Performed

- Python syntax checks passed for edited modules.
- Hybrid A* perpendicular smoke test succeeded on the default configuration:

```text
True Hybrid A*: OK in about 3.9s, 42300 iterations, 41 waypoints
```

## Known Notes

- The baseline MPC planner depends on SciPy. In the local environment used for validation, `scipy` was not installed.
- The Hybrid A* planner is currently an implementation slice, not yet a polished final planner. It still needs smoothing, metrics, and CARLA integration.

## Latest Update

Implemented the first parallel-parking algorithm:

- Generalized Hybrid A* collision checking from perpendicular-only rules to the union of lane and parking-spot rectangles.
- Added a parallel parking goal pose inside the curb-side spot.
- Routed `ParkingConfig(parking_type="parallel")` through Hybrid A* by default.
- Increased the Hybrid A* iteration cap so the default parallel scene can solve.
- Updated `README.md` to mark parallel parking as Hybrid A*-planned instead of unimplemented.

Refined parallel parking behavior:

- Slowed parallel-parking animation to one waypoint per frame.
- Tightened Hybrid A* success so a plan only succeeds when the final car rectangle is fully inside the target parking spot.
- Revalidated perpendicular and parallel Hybrid A* default paths with full-spot containment.

Continued development toward a stronger final-project deliverable:

- Added `metrics` to `TrajectoryResult`.
- Added Hybrid A* planner metrics: planning time, iterations, expanded states, path length, waypoint count, final pose error, full-spot containment, and obstacle count.
- Added `obstacle_scenario` to `ParkingConfig`.
- Added `scenarios.py` for static occupied regions in the parking lot.
- Added obstacle drawing in the settings preview and pygame simulation.
- Added a scenario selector to the settings UI.
- Added obstacle-aware planner dispatch: obstacle scenarios use Hybrid A* even if the requested planner is `baseline`.
- Added `evaluate.py` to print CSV metrics for clear and obstacle cases.
- Updated README with scenarios, metrics, and the evaluation command.

Documentation maintenance:

- Added `USER_GUIDE.md` with setup, settings, controls, planner modes, metrics, evaluation workflow, limitations, and suggested next improvements.
- Linked `USER_GUIDE.md` from `README.md`.
- Updated `PROJECT_SCOPE.md` with a staged roadmap for a stronger final demo: parameter sweeps, harder scenarios, algorithm comparison, smoothing, closed-loop tracking, visualization, and CARLA stretch integration.
