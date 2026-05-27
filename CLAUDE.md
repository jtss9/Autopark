# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
conda activate car
python main.py
```

Requires Python 3.11, `pygame>=2.0`, `scipy>=1.7`, `Pillow>=9.0` (see [requirements.txt](requirements.txt)).

There are no automated tests and no lint step — verify changes by running the app.

## Architecture

The app runs in two sequential phases, looping back on `S` keypress:

1. **Settings phase** — [settings_window.py](settings_window.py) opens a tkinter window with sliders. On confirm, it returns `(ParkingConfig, CarConfig)` to [main.py](main.py).
2. **Simulation phase** — [simulation.py](simulation.py) calls `plan_trajectory()`, then runs a pygame animation loop over the returned waypoints.

### Module responsibilities

| File | Role |
|---|---|
| [config.py](config.py) | `CarConfig` and `ParkingConfig` dataclasses — the only data passed between phases |
| [parking_lot.py](parking_lot.py) | `ParkingLot`: world-space geometry (`lane_rect`, `spot_rect`, `car_start_pose`, `car_corners`) |
| [trajectory.py](trajectory.py) | `plan_trajectory()` dispatcher → geometric reference planner → MPC planner(s) |
| [controller.py](controller.py) | `CarDynamics` (bicycle kinematic model) and `MPCController` (scipy SLSQP, horizon N=5, dt=0.05 s) |
| [simulation.py](simulation.py) | pygame render loop, HUD, collision overlay, keyboard handling |
| [settings_window.py](settings_window.py) | tkinter sliders + live preview canvas |

### Coordinate system

All world geometry uses **+x right, +y up** (metres). Both tkinter canvas and pygame flip y for rendering: `screen_y = origin_y − world_y × scale`. The reference point for every `Waypoint` and all corner calculations is the **rear axle centre**.

### Trajectory planning pipeline

`plan_trajectory()` in [trajectory.py](trajectory.py) dispatches based on `pc.planner`:

- **`"single"`** (`_plan_perpendicular_mpc`): builds a geometric 3-phase arc reference (drive forward → reverse arc → straight into spot), then runs `MPCController` tracking it. Fails with COLLISION in narrow lanes.
- **`"multi"`** (`_plan_perpendicular_multistep`): same arc reference per attempt, but uses `_arc_check()` as a forward-looking feasibility probe mid-arc. On WARN, executes a two-phase correction (Phase A: reverse with full right steer until |θ| < 2°; Phase B: scan forward to the largest x where `_arc_check` passes, then drive there). Repeats up to 5 times.

`MPCController.optimize()` uses scipy `minimize(method='SLSQP')` with a cost that penalises position error, heading error, steering effort, steering rate, and boundary violations (`w_boundary=2000`). Speed `v` is supplied by the caller so gear changes are handled in the planner, not the controller.

### Key constants

```
wheelbase       = car_length × 0.65
max_steer_angle = 35°  (math.radians)
R_min           = wheelbase / tan(max_steer_angle)   # ≈ 3–5 m
MPC horizon N   = 5,  dt = 0.05 s
```

### Parallel parking

`parking_type = "parallel"` is accepted by `ParkingConfig` and `ParkingLot._build_parallel()` builds the geometry, but `plan_trajectory()` returns an unimplemented `TrajectoryResult` immediately — no planner exists yet.
