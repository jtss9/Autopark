# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
conda activate car
pip install -r requirements.txt
python main.py
```

Requires Python 3.11, `pygame>=2.0`, `Pillow>=9.0`, `scipy>=1.7` (which pulls in NumPy; `controller.py` imports it). See [requirements.txt](requirements.txt).

Optional environment overrides (read in [simulation.py](simulation.py)):

```bash
AUTOPARK_PLANNER=hybrid_astar python main.py   # force a planner, overriding the Settings UI
AUTOPARK_TRACK=1 python main.py                # run the Pure Pursuit tracker and overlay the executed path
```

`AUTOPARK_PLANNER` accepts `single` | `multi` | `hybrid_astar` | `qlearn` (the legacy alias `baseline` maps to `single`). Leave it unset to honour the Settings-UI choice.

There is no lint step and no unit-test suite. Verify behaviour by running the app or the headless evaluator (below).

### Headless evaluation

[evaluate.py](evaluate.py) is the regression/benchmark harness — it runs planners across scenarios and lane/car sweeps and writes a CSV; [plot_results.py](plot_results.py) renders report figures from those CSVs.

```bash
python evaluate.py                                                   # default run
python evaluate.py --mode all --scenario all --planner all --output results/all.csv
python evaluate.py --planner hybrid_astar --track --output results/tracked.csv
python plot_results.py results/all.csv
```

CLI flags: `--mode`, `--scenario`, `--sweep`, `--planner`, `--track`, `--output`.

## Architecture

The interactive app runs in two sequential phases, looping back on the `S` keypress:

1. **Settings phase** — [settings_window.py](settings_window.py) opens a tkinter window with sliders + radio buttons and a live preview canvas. On confirm it returns `(ParkingConfig, CarConfig)` to [main.py](main.py).
2. **Simulation phase** — [simulation.py](simulation.py) calls `plan_trajectory()`, then runs a pygame animation loop over the returned waypoints (and the closed-loop executed path when tracking is on).

### Module responsibilities

| File | Role |
|---|---|
| [config.py](config.py) | `CarConfig` and `ParkingConfig` dataclasses — the only data passed between phases |
| [parking_lot.py](parking_lot.py) | `ParkingLot`: world geometry (`lane_rect`, `spot_rect`, `car_start_pose`, `car_corners`) |
| [geom.py](geom.py) | Shared helpers: `angle_diff`, `wrap_pi`, `split_by_gear`, `path_length` (centralised to avoid per-module drift) |
| [trajectory.py](trajectory.py) | `plan_trajectory()` dispatcher + the two MPC planners + tracker attachment + metrics |
| [controller.py](controller.py) | `CarDynamics` (bicycle kinematic model) and `MPCController` (scipy SLSQP, horizon N=5, dt=0.05 s) |
| [hybrid_astar.py](hybrid_astar.py) | `plan_hybrid_astar()`: state-lattice A* over (x, y, θ) with a Reeds-Shepp analytic shot |
| [reeds_shepp.py](reeds_shepp.py) | Reeds-Shepp shortest paths: `shortest_path`, `discretize`, `path_length` (used as the A* heuristic + analytic shot) |
| [tracker.py](tracker.py) | `track_path()`: Pure Pursuit closed-loop tracker over gear-split segments; `TrackerConfig`, `TrackResult` |
| [rl_qlearn.py](rl_qlearn.py) | `plan_qlearn()`: tabular Q-learning parking baseline |
| [scenarios.py](scenarios.py) | `obstacles_for(lot)` → list of `Rect` obstacles for the selected obstacle scenario |
| [simulation.py](simulation.py) | pygame render loop, HUD, occupancy-grid overlay, executed-path overlay, keyboard handling |
| [settings_window.py](settings_window.py) | tkinter sliders/radios + live preview canvas |
| [evaluate.py](evaluate.py) | Headless benchmark harness → CSV |
| [plot_results.py](plot_results.py) | Render report figures from evaluator CSVs |
| [carla_bridge.py](carla_bridge.py) · [carla_controller.py](carla_controller.py) · [carla_demo.py](carla_demo.py) | CARLA stretch goal (see below) |

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
| parallel | `plan_hybrid_astar` |

When `track=True` and the plan is feasible, `_attach_tracker` densifies the path and runs `tracker.track_path`, populating `result.executed_waypoints` and `result.tracking_metrics` (mean/max CTE, final pose error, fully-in-spot, cusps).

### Planners

- **Single-step MPC** (`_plan_perpendicular_mpc`): builds a geometric 3-phase arc reference (forward → reverse arc → straight into spot) and tracks it with `MPCController`. Reactive boundary penalty keeps it in bounds; fails (collision) in narrow lanes.
- **Multi-step MPC** (`_plan_perpendicular_multistep`): a narrow-lane specialist. Each attempt builds a fresh arc reference from the car's current pose, reverses to the goal depth (`spot_top − 0.15`), and on a road-bottom WARN runs a two-phase correction (reverse hard-right until |θ|≈0, then drive forward to a safe x), repeating up to 5 times. Uses a **rigid** arc, so for a large car in ~5 m lanes it can clip the spot edge where single-step's reactive MPC succeeds.
- **Hybrid A\*** (`plan_hybrid_astar`): state-lattice A* over discretised (x, y, θ) with forward/reverse × 5 steering primitives, a Reeds-Shepp analytic shot (attempted every `rs_shot_interval=50` expansions and unconditionally within `rs_shot_radius=4.5` m), an RS-length heuristic cache, and final path smoothing. Handles obstacles and parallel parking. Key tunables live on `HybridAStarPlanner.__init__` (`xy_resolution=0.10`, `theta_bins=36`, `motion_step=0.45`, `goal_xy_tolerance=0.35`, `max_iterations=150000`).
- **Q-learning** (`plan_qlearn`): tabular RL baseline, kept for comparison in the evaluator.

`MPCController.optimize()` uses scipy `minimize(method='SLSQP')` with a cost over position error, heading error, steering effort, steering rate, and boundary violation (`w_boundary`). Speed `v` is supplied by the caller, so gear changes are owned by the planner, not the controller.

### Key constants

```
wheelbase        = car_length × 0.65
max_steer        = 35°
min_turn_radius  = wheelbase / tan(max_steer)        # property on CarConfig, ≈ 3–5 m
MPC horizon N    = 5,  dt = 0.05 s
```

### Simulation keys

`SPACE` pause/resume · `R` restart animation · `S` back to settings · `G` toggle occupancy-grid overlay · `T` toggle executed (tracker) path overlay · `ESC`/`Q` quit.

### Obstacle scenarios

`ParkingConfig.obstacle_scenario` ∈ `none` · `entry_blocker` · `tight_lane` · `pillar_near_entry` · `parked_cars`. `scenarios.obstacles_for(lot)` returns the corresponding `Rect` obstacles (empty for `none`/`tight_lane`). Any non-`none` scenario forces the Hybrid A* backend.

### CARLA stretch goal

An optional bridge to the CARLA simulator (requires the `carla` Python package):

- [carla_bridge.py](carla_bridge.py) — thin adapter between this planner/controller stack and a CARLA world (pose/frame transforms, static-obstacle extraction around the ego).
- [carla_controller.py](carla_controller.py) — `CarlaPurePursuitController`: same Pure Pursuit logic as `tracker.py`, emitting CARLA `VehicleControl` (steer/throttle/brake).
- [carla_demo.py](carla_demo.py) — end-to-end demo with a `--dry-run` integrator path (no CARLA needed) and `--ctl-*` / `--max-seconds` tuning flags.
