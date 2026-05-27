# Project Scope

## Working Title

GPS-denied autonomous parking in structured parking lots using Hybrid A* planning and MPC control.

Chinese title:

基於 Hybrid A* 與 MPC 的無 GPS 停車場自主泊車系統

## Motivation

The original proposal focused on indoor parking-lot SLAM mapping:

> 停車場室內 SLAM 建圖：在無 GPS 的地下或室內停車場環境中，應用 LiDAR 或 RGB-D SLAM 技術建立地圖，實現自主停車導航。

For the final project, the stronger and more feasible focus is autonomous parking rather than building a complete SLAM stack from scratch. The map can come from a prebuilt parking-lot map, a simulator ground-truth map, or a SLAM-generated occupancy grid. The project then focuses on planning, collision checking, and closed-loop vehicle control.

## Current Baseline

The current simulator implements a fixed perpendicular parking maneuver:

1. Build a geometric reference path.
2. Drive forward beside the target spot.
3. Reverse through a fixed 90-degree arc.
4. Reverse straight into the spot.
5. Track the reference with a short-horizon MPC controller.

This is useful as a baseline, but it is too constrained for a master-level final project because it assumes a simple scene, a fixed maneuver, and no arbitrary obstacles.

## Target System

```text
Known map / SLAM map / CARLA scene
        |
Occupancy grid or costmap
        |
Parking goal pose selection
        |
Hybrid A* parking planner
        |
Path smoothing
        |
MPC / path-tracking controller
        |
Closed-loop simulation and evaluation
```

## Implementation Plan

1. Keep the existing geometric-arc MPC planner as a baseline.
2. Add an occupancy grid abstraction for parking-lot free space.
3. Implement Hybrid A* over state `(x, y, theta)`.
4. Use full car-rectangle collision checking against lane, spot, and obstacles.
5. Support perpendicular first, then extend to parallel and arbitrary goal poses.
6. Add metrics: planning time, path length, final pose error, collision count, and success rate.
7. Integrate with CARLA if time allows, using CARLA as the realistic simulator and map/sensor provider.

## Roadmap Toward a Strong Final Demo

### Stage 1: Measurable Local Planner

Goal: turn the simulator from a visual demo into an experiment platform.

- Expand `evaluate.py` from four fixed cases into parameter sweeps.
- Sweep lane width, spot size, car size, and obstacle position.
- Export CSV results for report tables and plots.
- Report success rate, planning time, path length, final pose error, heading error, and full-spot containment.

### Stage 2: Harder Parking Scenarios

Goal: demonstrate that the planner handles cases beyond a fixed textbook maneuver.

- Add multiple parked cars.
- Add narrow-lane scenes.
- Add a pillar near the parking-spot entrance.
- Add partially blocked target spots.
- Add random static obstacle maps.
- Show failure reasons when the scene is physically infeasible.

### Stage 3: Algorithm Comparison

Goal: make the final report look like an autonomy/planning project, not only an implementation demo.

- Compare the baseline geometric/MPC planner against Hybrid A*.
- Add another planner if time allows, such as RRT* or a lattice planner.
- Evaluate each planner on the same scenario set.
- Report where each planner succeeds, fails, or becomes slow.

### Stage 4: Path Smoothing

Goal: make Hybrid A* output more vehicle-like and less grid-like.

- Add a smoothing pass after Hybrid A*.
- Penalize curvature, steering changes, and collision violations.
- Compare raw vs smoothed paths.
- Report path length, heading smoothness, and steering smoothness.

### Stage 5: Closed-Loop Tracking

Goal: separate planning success from control success.

- Track the planned path with Pure Pursuit as a simple controller.
- Reuse or extend MPC for trajectory tracking.
- Measure tracking error over time.
- Display planned path vs executed path.
- Report maximum tracking error and final parking error.

### Stage 6: Better Visualization

Goal: make the demo explain the algorithm as it runs.

- Draw the occupancy grid.
- Draw static obstacles as occupied cells.
- Draw expanded Hybrid A* states or search tree samples.
- Draw planned path and executed path separately.
- Show planner metrics in the HUD.
- Add success/failure labels with concrete reasons.

### Stage 7: CARLA Stretch Goal

Goal: connect the local planner to a realistic simulator.

- Build or select a CARLA parking-lot scene.
- Spawn ego vehicle and parked vehicles.
- Use CARLA ground-truth map or generated occupancy grid first.
- Later, derive local obstacles from LiDAR/depth if time allows.
- Feed the same Hybrid A* planner with CARLA map/obstacle data.
- Execute the resulting path in CARLA with a tracking controller.

## Target Final Demo

The final demo should let the user choose:

```text
Parking mode: perpendicular / parallel
Scenario: clear / tight / obstacle / random
Planner: baseline / Hybrid A*
```

The system should then:

1. Generate or load the parking map.
2. Build an occupancy grid.
3. Plan a collision-free parking path.
4. Animate or execute the vehicle.
5. Report quantitative metrics.
6. Clearly state success or failure.

The intended final claim:

> We implemented and evaluated an autonomous parking pipeline using occupancy-grid Hybrid A*, full vehicle collision checking, strict parking success criteria, and quantitative benchmarking across structured parking scenarios.

## Near-Term Development Slice

The first implementation slice is local and testable inside the current Python simulator:

1. Create a `hybrid_astar.py` planner module.
2. Generate a grid from `ParkingLot` geometry.
3. Plan from `lot.car_start_pose` to a goal pose inside the parking spot.
4. Return the same `TrajectoryResult` / `Waypoint` shape used by the current simulation.
5. Keep the existing planner available as the default until Hybrid A* is stable.

## Evaluation Goals

- The planner reaches a valid parking pose without collision.
- The planner handles tighter lanes better than the fixed arc baseline.
- The planner can react to blocked cells or parked-car obstacles.
- The project can compare baseline geometric MPC vs. Hybrid A* + controller.
- The project reports quantitative metrics instead of relying only on visual inspection.
- The final vehicle rectangle must be fully inside the target parking spot for success.

## Two-Week Autonomous Parking Roadmap

### Summary

Build toward a stronger master-coursework deliverable by turning the current simulator into a measurable autonomous parking experiment platform, while keeping CARLA as a stretch goal. The one-week checkpoint should show working progress plus a live demo: Hybrid A* perpendicular and parallel parking, obstacle scenarios, metrics, and a clear report roadmap. The two-week report should include parameter-sweep evaluation, harder scenarios, visualization, and at least one path-quality improvement.

Assumption: since no focus option was selected, use evaluation first as the default, with enough demo polish for presentation.

### Week 1: Presentation-Ready Progress

#### Evaluation Upgrade

- Expand `evaluate.py` from four fixed cases into a configurable evaluator.
- Add CLI options:
  - `--mode perpendicular|parallel|all`
  - `--scenario clear|obstacle|tight|all`
  - `--sweep lane_width|spot_size|car_size|none`
  - `--output path.csv`
- Keep stdout CSV behavior when `--output` is omitted.
- Add aggregate summary rows or printed summary:
  - success rate
  - average planning time
  - average path length
  - average final position error
  - full-spot success rate

#### Scenario Upgrade

- Extend `scenarios.py` with named scenarios:
  - `none`
  - `entry_blocker`
  - `tight_lane`
  - `pillar_near_entry`
  - `parked_cars`
- Update `ParkingConfig.obstacle_scenario` to accept those names.
- Update the settings UI scenario selector to expose at least:
  - Clear
  - Entry Blocker
  - Tight Lane
  - Parked Cars
- Draw all obstacle rectangles in preview and simulation.

#### Demo Polish

- Improve HUD metrics display:
  - path length
  - planning time
  - final error
  - full-in-spot result
  - success/failure
- Add clear failure messages for:
  - start pose invalid
  - goal pose invalid
  - no path found
  - final car not fully inside spot
- Update `USER_GUIDE.md` with the new evaluator options and scenarios.

#### Week 1 Acceptance Criteria

- Presentation can show:
  - perpendicular clear run
  - parallel clear run
  - obstacle scenario run
  - `python evaluate.py --mode all --scenario all`
- All successful cases report `fully_in_spot=True`.
- README/User Guide explain how to reproduce the demo.
- `RECORD.md` is updated with what was completed.

### Week 2: Report-Strengthening Work

#### Parameter Sweeps

- Add sweep generators for:
  - lane width: `3.5, 4.0, 4.5, 5.0, 5.5`
  - spot width: `2.0, 2.3, 2.5, 2.8, 3.0`
  - spot length: `5.0, 5.5, 6.0`
  - car length: `3.5, 4.0, 4.5, 5.0`
- Each evaluation row must include:
  - parking mode
  - scenario
  - lane width
  - spot length
  - spot width
  - car length
  - car width
  - feasible
  - planning metrics
  - final pose metrics
- Add `results/` output support for CSV files, but do not require checked-in generated results unless intentionally added for the report.

#### Algorithm Comparison

- Evaluate baseline where applicable:
  - baseline only for perpendicular clear cases
  - Hybrid A* for perpendicular, parallel, and obstacle cases
- Add a `planner` column to evaluation outputs.
- If SciPy is missing, baseline rows should be skipped with a clear reason rather than crashing.
- Report comparison:
  - baseline is faster and simple but limited
  - Hybrid A* is slower but handles parallel parking and obstacles

#### Path Quality Improvement

- Add a lightweight smoothing/post-processing step for Hybrid A* output.
- Minimum acceptable smoothing:
  - remove redundant collinear or near-duplicate waypoints
  - preserve first and final waypoint
  - reject smoothed segments that collide or leave lane/spot bounds
- Add metrics:
  - raw path length
  - smoothed path length
  - raw waypoint count
  - smoothed waypoint count
- Keep smoothing optional behind a planner option or internal default for Hybrid A* only.

#### Visualization Improvement

- Add optional drawing of:
  - obstacle rectangles
  - planned path
  - smoothed path if available
  - final success/failure label
- If feasible within time, add sampled Hybrid A* expanded states, capped to avoid slow rendering.

#### Week 2 Acceptance Criteria

- `python evaluate.py --mode all --scenario all --sweep lane_width --output results/lane_width.csv` works.
- At least one CSV table is usable directly in the report.
- Report can include:
  - success-rate table
  - planning-time comparison
  - path-length comparison
  - baseline vs. Hybrid A* limitations
- Demo can show clear, obstacle, and tight scenarios.
- Final docs include limitations and CARLA stretch discussion.

### Interfaces And Data Changes

- Keep `ParkingConfig` as the main scenario/config input.
- Extend `obstacle_scenario` values to:
  - `none`
  - `entry_blocker`
  - `tight_lane`
  - `pillar_near_entry`
  - `parked_cars`
- Keep `TrajectoryResult.metrics` as a dictionary, but standardize keys:
  - `planning_time_s`
  - `iterations`
  - `expanded_states`
  - `path_length_m`
  - `waypoints`
  - `final_pos_error_m`
  - `final_heading_error_deg`
  - `fully_in_spot`
  - `obstacles`
  - optional: `raw_path_length_m`, `smoothed_path_length_m`
- `evaluate.py` becomes the report-facing CLI. It should never require the GUI.

### Test Plan

- Syntax:
  - `python3 -m py_compile config.py parking_lot.py scenarios.py trajectory.py hybrid_astar.py simulation.py settings_window.py evaluate.py`
- Planner smoke tests:
  - perpendicular + clear
  - perpendicular + entry blocker
  - parallel + clear
  - parallel + entry blocker
  - tight lane expected success/failure depending on dimensions
- Evaluation tests:
  - `python evaluate.py`
  - `python evaluate.py --mode parallel --scenario all`
  - `python evaluate.py --mode all --scenario all --sweep lane_width`
- Success assertions:
  - no crash on infeasible cases
  - failed cases include a reason
  - successful cases have `fully_in_spot=True`
  - CSV columns remain stable
- Manual demo tests:
  - settings selector updates preview
  - simulation draws obstacles
  - HUD shows metrics
  - `S`, `R`, `SPACE`, `ESC/Q` still work

### Assumptions

- CARLA is stretch only for this two-week plan.
- Week 1 presentation should emphasize both progress and live demo.
- Primary report strength comes from quantitative evaluation, not only visual animation.
- Baseline remains useful as a comparison, but Hybrid A* is the main planner.
- Path smoothing should be lightweight and collision-checked; full optimal-control smoothing is out of scope unless time remains.
