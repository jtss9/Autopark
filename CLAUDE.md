# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project layout

`main.py` is the **only** Python file in the repository root. Every other source
module lives in [src/](src/). `main.py` prepends `src/` to `sys.path` at startup,
so all modules keep flat imports (`from config import ...`). When running the
helper scripts directly (`python src/evaluate.py`), Python adds `src/` to the
path automatically because the script lives there.

## Running the App

```bash
conda activate car
pip install -r requirements.txt
python main.py
```

Requires Python 3.11, `pygame>=2.0`, `Pillow>=9.0`, `scipy>=1.7` (which pulls in NumPy; `controller.py` imports it). See [requirements.txt](requirements.txt).

Optional environment overrides (read in [src/simulation.py](src/simulation.py)):

```bash
AUTOPARK_PLANNER=hybrid_astar python main.py   # force a planner, overriding the Settings UI
AUTOPARK_TRACK=1 python main.py                # run the Pure Pursuit tracker and overlay the executed path
```

`AUTOPARK_PLANNER` accepts `single` | `multi` | `hybrid_astar` | `qlearn` (the legacy alias `baseline` maps to `single`). Leave it unset to honour the Settings-UI choice. Note: `multi` is still selectable here and in the evaluator, but is **hidden in the Settings UI** (see Planners below).

There is no lint step and no unit-test suite. Verify behaviour by running the app or the headless evaluator (below).

### Headless evaluation

[src/evaluate.py](src/evaluate.py) is the regression/benchmark harness — it runs planners across scenarios and lane/car sweeps and writes a CSV; [src/plot_results.py](src/plot_results.py) renders report figures from those CSVs.

```bash
python src/evaluate.py                                                   # default run
python src/evaluate.py --mode all --scenario all --planner all --output results/all.csv
python src/evaluate.py --planner hybrid_astar --track --output results/tracked.csv
python src/plot_results.py results/all.csv
```

CLI flags: `--mode`, `--scenario`, `--sweep`, `--planner`, `--track`, `--output`.

## Architecture

The interactive app runs in two sequential phases, looping back on the `S` keypress:

1. **Settings phase** — [src/settings_window.py](src/settings_window.py) opens a tkinter window with sliders, parking-type / planner toggle buttons, and a live preview canvas. When the planner is **Hybrid A***, an "Add obstacle" checkbox appears and a fixed-size obstacle can be dragged on the preview (kept off the car and the spot). On confirm it returns `(ParkingConfig, CarConfig)` to [main.py](main.py).
2. **Simulation phase** — [src/simulation.py](src/simulation.py) calls `plan_trajectory()`, then runs a pygame animation loop over the returned waypoints (and the closed-loop executed path when tracking is on).

### Module responsibilities

All modules live under [src/](src/) (`main.py` is the only root file).

| File | Role |
|---|---|
| [config.py](src/config.py) | `CarConfig` and `ParkingConfig` dataclasses — the only data passed between phases. `ParkingConfig.obstacle = (x, y, w, h) \| None` carries the user-placed obstacle |
| [parking_lot.py](src/parking_lot.py) | `ParkingLot`: world geometry (`lane_rect`, `spot_rect`, `car_start_pose`, `car_corners`); `Rect` |
| [geom.py](src/geom.py) | Shared helpers: `angle_diff`, `wrap_pi`, `split_by_gear`, `path_length` (centralised to avoid per-module drift) |
| [trajectory.py](src/trajectory.py) | `plan_trajectory()` dispatcher + the perpendicular/parallel MPC planners + early-stop logic + tracker attachment + metrics |
| [controller.py](src/controller.py) | `CarDynamics` (bicycle kinematic model) and `MPCController` (scipy SLSQP, horizon N=5, dt=0.05 s) |
| [hybrid_astar.py](src/hybrid_astar.py) | `plan_hybrid_astar()`: state-lattice A* over (x, y, θ) with a Reeds-Shepp analytic shot; `OccupancyGrid` with corner + SAT body collision checks |
| [reeds_shepp.py](src/reeds_shepp.py) | Reeds-Shepp shortest paths: `shortest_path`, `discretize`, `path_length` (used as the A* heuristic + analytic shot) |
| [tracker.py](src/tracker.py) | `track_path()`: Pure Pursuit closed-loop tracker over gear-split segments; `TrackerConfig`, `TrackResult` |
| [rl_qlearn.py](src/rl_qlearn.py) | `plan_qlearn()`: tabular Q-learning parking baseline |
| [scenarios.py](src/scenarios.py) | `obstacles_for(lot)` → list of `Rect` obstacles for a named scenario (used by the evaluator; not exposed in the interactive UI) |
| [simulation.py](src/simulation.py) | pygame render loop, HUD, occupancy-grid overlay, executed-path overlay, keyboard handling |
| [settings_window.py](src/settings_window.py) | tkinter sliders/toggle buttons + live preview canvas + draggable obstacle |
| [evaluate.py](src/evaluate.py) | Headless benchmark harness → CSV |
| [plot_results.py](src/plot_results.py) | Render report figures from evaluator CSVs |
| [carla_bridge.py](src/carla_bridge.py) · [carla_controller.py](src/carla_controller.py) · [carla_demo.py](src/carla_demo.py) | CARLA stretch goal (see below) |

### Coordinate system

All world geometry uses **+x right, +y up** (metres), θ in radians. Both tkinter and pygame flip y for rendering (`screen_y = origin_y − world_y × scale`). The reference point for every `Waypoint` and all corner calculations is the **rear-axle centre** (`ParkingLot.car_corners`).

### Trajectory planning pipeline

`plan_trajectory(pc, cc, planner=None, track=False)` resolves the backend as `effective = planner if planner is not None else pc.planner` (explicit caller intent always wins; the Settings UI is the fallback), then dispatches:

| Condition | Backend |
|---|---|
| `effective == "qlearn"` | `rl_qlearn.plan_qlearn` |
| `effective == "hybrid_astar"` | `hybrid_astar.plan_hybrid_astar` |
| `pc.obstacle_scenario != "none"` | `plan_hybrid_astar` (auto-promoted — the MPC planners have no obstacle awareness) |
| perpendicular + `effective == "multi"` | `_plan_perpendicular_multistep` |
| perpendicular (otherwise) | `_plan_perpendicular_mpc` (single-step) |
| parallel | `_plan_parallel_mpc` |

The user-placed obstacle (`pc.obstacle`) is consumed by `plan_hybrid_astar` and by the SAC planner (`rl_sac.plan_sac` → `ParkingEnv`, whose observation carries nearest-obstacle features); these are also the only planners exposed with the obstacle checkbox, so the MPC planners never see an obstacle they cannot reason about.

When `track=True` and the plan is feasible, `_attach_tracker` densifies the path and runs `tracker.track_path`, populating `result.executed_waypoints` and `result.tracking_metrics` (mean/max CTE, final pose error, fully-in-spot, cusps).

### Planners

- **Single-step MPC** (`_plan_perpendicular_mpc`): builds a geometric 3-phase arc reference (forward → reverse arc → straight into spot) and tracks it with `MPCController`. Reactive boundary penalty keeps it in bounds; fails (collision) in narrow lanes.
- **Parallel MPC** (`_plan_parallel_mpc`): the parallel-parking counterpart. `_plan_parallel` builds an equal-radius two-arc S-curve reference (right-steer reverse → left-steer reverse) from `Δy = lane_width/2 + spot_width/2` and `α = arccos(1 − Δy/2R)`; `MPCController` tracks it into the curb-side spot.
- **Multi-step MPC** (`_plan_perpendicular_multistep`): a narrow-lane specialist that rebuilds a fresh arc each attempt with a reverse/forward correction. Still callable via `AUTOPARK_PLANNER=multi` and the evaluator, but **hidden from the Settings UI** (it does not generalise cleanly across sizes).
- **Hybrid A\*** (`plan_hybrid_astar`): state-lattice A* over discretised (x, y, θ) with forward/reverse × 5 steering primitives, a Reeds-Shepp analytic shot (attempted every `rs_shot_interval=50` expansions and unconditionally within `rs_shot_radius=4.5` m), an RS-length heuristic cache, and final path smoothing. Handles obstacles and parallel parking. Key tunables live on `HybridAStarPlanner.__init__` (`xy_resolution=0.10`, `theta_bins=36`, `motion_step=0.45`, `goal_xy_tolerance=0.35`, `max_iterations=150000`).
- **Q-learning** (`plan_qlearn`): tabular RL baseline, kept for comparison in the evaluator.

**Early-stop ("park then align").** Both MPC planners (`_plan_perpendicular_mpc`, `_plan_parallel_mpc`) stop as soon as the whole car body is inside the spot (`_pose_in_spot`) **and** the heading is within `ALIGN_TOL = 8°` of the goal, instead of finishing the reference path (which would keep reversing and clip the spot's far edge). They track the most-aligned in-spot pose (`best_idx`); if a later step would leave bounds, they truncate back to that pose and report success rather than a collision.

`MPCController.optimize()` uses scipy `minimize(method='SLSQP')` with a cost over position error, heading error, steering effort, steering rate, and boundary violation (`w_boundary`). Speed `v` is supplied by the caller, so gear changes are owned by the planner, not the controller.

### Key constants

```
wheelbase        = car_length × 0.65
max_steer        = 35°
min_turn_radius  = wheelbase / tan(max_steer)        # property on CarConfig, ≈ 3–5 m
MPC horizon N    = 5,  dt = 0.05 s
ALIGN_TOL        = 8°  (MPC early-stop heading tolerance)
```

### Simulation keys

`SPACE` pause/resume · `R` restart animation · `S` back to settings · `G` toggle occupancy-grid overlay · `T` toggle executed (tracker) path overlay · `↑`/`↓` animation speed · `ESC`/`Q` quit.

### Obstacles

There are two distinct obstacle paths:

- **Interactive (UI).** When the planner is Hybrid A*, the settings window lets you drop one fixed-size obstacle (`SettingsWindow.OBS_W/OBS_H`, 0.9 m) and drag it on the preview. It is stored as `ParkingConfig.obstacle = (x, y, w, h)`, drawn in the simulation, and fed into `plan_hybrid_astar`. The UI forbids placing it on the car (start) or the spot (goal). `OccupancyGrid.pose_is_valid` checks it with both the rasterised corner test and an exact **SAT** car-body-vs-obstacle intersection (`_obb_hits_rect`, `obstacle_margin = 0.05`), so a small obstacle under the car's middle is not missed.
- **Named scenarios (evaluator).** `ParkingConfig.obstacle_scenario` ∈ `none` · `entry_blocker` · `tight_lane` · `pillar_near_entry` · `parked_cars`, expanded by `scenarios.obstacles_for(lot)`. Used by `evaluate.py`; any non-`none` scenario forces the Hybrid A* backend. Not exposed in the interactive UI.

### CARLA stretch goal

An optional bridge to the CARLA simulator (requires the `carla` Python package):

- [src/carla_bridge.py](src/carla_bridge.py) — thin adapter between this planner/controller stack and a CARLA world (pose/frame transforms, static-obstacle extraction around the ego).
- [src/carla_controller.py](src/carla_controller.py) — `CarlaPurePursuitController`: same Pure Pursuit logic as `tracker.py`, emitting CARLA `VehicleControl` (steer/throttle/brake).
- [src/carla_demo.py](src/carla_demo.py) — end-to-end demo with a `--dry-run` integrator path (no CARLA needed) and `--ctl-*` / `--max-seconds` tuning flags.
