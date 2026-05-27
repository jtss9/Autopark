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

## Latest Main Merge And Evaluation Upgrade

Merged the latest `origin/main` into `mason-implement`:

- Main added `ParkingConfig.planner`.
- Main added the multi-step MPC planner for narrow-lane perpendicular parking.
- The merge resolution preserved both main's `planner` field and this branch's `obstacle_scenario` field.
- The settings window now exposes Single-step MPC, Multi-step MPC, and Hybrid A*.

Expanded the report-facing evaluator:

- Added CLI options for `--mode`, `--scenario`, `--planner`, `--sweep`, and `--output`.
- Added stable CSV columns for mode, scenario, planner, dimensions, feasibility, success, planning metrics, final-pose metrics, and full-spot containment.
- Added baseline comparison rows where applicable.
- Added clean skip rows when the baseline cannot run, including when SciPy is unavailable.
- Added output-file support with automatic parent directory creation.
- Kept stdout CSV behavior when `--output` is omitted, with the aggregate summary written to stderr.

Expanded scenarios:

- Added canonical scenario names: `none`, `entry_blocker`, `tight_lane`, `pillar_near_entry`, and `parked_cars`.
- Added aliases for evaluator convenience: `clear`, `obstacle`, and `tight`.
- Added obstacle rectangles for entry blockers, pillars, and parked cars.

Added lightweight Hybrid A* path cleanup:

- Removes near-duplicate and nearly collinear waypoints.
- Preserves the first and final waypoint.
- Rejects cleanup segments that collide or leave valid lane/spot bounds.
- Reports raw and smoothed path lengths and waypoint counts.

Validation:

- `python3 -m py_compile config.py parking_lot.py scenarios.py trajectory.py hybrid_astar.py simulation.py settings_window.py evaluate.py`
- `python3 evaluate.py --mode perpendicular --scenario clear --planner all`
- `python3 evaluate.py --mode parallel --scenario all --planner hybrid_astar --output /tmp/autopark_parallel.csv`
- `python3 evaluate.py --mode perpendicular --scenario all --planner hybrid_astar --output /tmp/autopark_perpendicular.csv`

Observed validation notes:

- The local Python environment used for validation does not have SciPy, so baseline rows are skipped cleanly.
- Parallel all-scenario Hybrid A* smoke test succeeded for all five scenarios.
- Perpendicular all-scenario Hybrid A* smoke test succeeded for four of five scenarios; `tight_lane` correctly produced an infeasible/no-path row at the current default dimensions.

## HUD And Baseline Metrics Update

Continued Week 1 demo polish:

- Added shared trajectory metrics for MPC-based planners, not only Hybrid A*.
- Standardized MPC result metrics with `planning_time_s`, `iterations`, `expanded_states`, `path_length_m`, `waypoints`, final pose error, heading error, `fully_in_spot`, and obstacle count.
- Tightened MPC success so a nominally feasible plan is marked failed if the final car rectangle is not fully inside the spot.
- Updated the pygame HUD to always show status, phase, step, path length, planning time, final position error, heading error, and full-spot containment when metrics are available.
- Replaced the failure overlay text with `FAILED`, because failures can be collision, no path, timeout, or final-containment failure.
