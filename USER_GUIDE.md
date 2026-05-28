# User Guide

## 1. Purpose

This simulator demonstrates autonomous parking in a structured parking-lot scene.
It supports:

- Reverse perpendicular parking and parallel parking.
- Clear and obstacle scenarios (entry blocker, pillar, parked cars, tight lane).
- A baseline geometric / MPC planner for simple perpendicular cases.
- A Hybrid A* planner with Reeds-Shepp analytic shot for map-based,
  obstacle-aware planning over `(x, y, θ)`.
- A tabular Q-learning RL planner as a learned-policy comparison baseline.
- A Pure Pursuit closed-loop tracker that re-executes the planned path under
  the kinematic bicycle model and reports cross-track tracking error.
- A CSV evaluator and matplotlib plot generator for final-project reporting.

## 2. Setup

Install dependencies (the conda env `car` is what the project assumes; the
base conda env also works as long as the same packages are installed):

```bash
conda activate car             # or: conda activate base
pip install -r requirements.txt
pip install matplotlib         # needed by plot_results.py
```

Run the app:

```bash
python main.py
```

Pick a planner via environment variable (overrides the Settings UI choice):

```bash
AUTOPARK_PLANNER=hybrid_astar python main.py
AUTOPARK_PLANNER=qlearn python main.py
```

Run the simulator with the Pure Pursuit tracker overlay enabled:

```bash
AUTOPARK_TRACK=1 AUTOPARK_PLANNER=hybrid_astar python main.py
```

Run batch evaluation:

```bash
python evaluate.py
python evaluate.py --mode all --scenario all --planner all --output results/main.csv
python evaluate.py --planner hybrid_astar --track --output results/tracked.csv
python evaluate.py --mode all --scenario all --sweep lane_width --planner hybrid_astar --output results/sweep_lane_width.csv
```

Render report figures from a CSV:

```bash
python plot_results.py results/main.csv --outdir results/figures
```

## 3. Settings Window

The settings window is the first screen. It lets you configure the parking
environment and vehicle.

### Dimensions

Use the sliders to adjust:

- Lane width.
- Parking spot length.
- Parking spot width.
- Car length.
- Car width.

The preview updates immediately when a slider changes. If the car cannot fit
inside the selected parking spot, the spot outline turns red and the simulation
cannot start.

### Parking Type

- **Reverse into Spot** — perpendicular reverse parking.
- **Parallel Parking** — curb-side parallel parking.

### Scenario

- **Clear** — no extra obstacle.
- **Entry Blocker** — adds one occupied region near the spot entry.
- **Tight Lane** — narrow-lane evaluator scenario.
- **Parked Cars** — adds parked-car obstacle rectangles around the target.

Obstacle scenarios automatically use Hybrid A* when the selected planner does
not natively reason over occupancy obstacles.

### Planner

- **Single-step MPC** — tracks a pre-computed geometric arc. Wide-lane only.
- **Multi-step MPC** — alternating reverse/correction attempts; extends
  feasibility into narrower lanes.
- **Hybrid A*** — `(x, y, θ)` search with full car-corner collision checking,
  Reeds-Shepp analytic shot, RS-based heuristic, and lightweight smoothing.
- **Q-learning (RL)** — tabular Q-learning with reverse-curriculum training
  over a discretized state grid. Included as a learned-policy comparison.

## 4. Simulation Window

After clicking **Start Simulation**, the pygame window opens (1080 × 720).

### Controls

| Key | Action |
|---|---|
| `SPACE` | Pause or resume animation |
| `R` | Restart current animation |
| `G` | Toggle occupancy-grid overlay |
| `T` | Toggle executed (closed-loop) path overlay |
| `S` | Return to settings window |
| `ESC` / `Q` | Quit |

### HUD Fields

The HUD shows:

- Parking type, active planner, scenario.
- Lane, spot, and car dimensions; minimum turning radius.
- Status (`SUCCESS` / `FAILED`), current phase name, step counter.
- Path length, planning time, final position / heading error, full-spot
  containment.
- When Hybrid A* used the Reeds-Shepp shot: success / attempt count.
- When the planner is Q-learning: training time, successful episodes,
  Q-table size.
- When the tracker ran (`AUTOPARK_TRACK=1`): mean / max cross-track error,
  number of cusps, executed final pose error, executed in-spot flag.

### What Step Means

`Step` is the current waypoint index in the planned trajectory.

```text
Step: 20 / 36
```

The animation is showing waypoint 20 out of 36 — not seconds or real time.
Parallel parking is animated slower, advancing one waypoint per frame.

## 5. Planners

### Baseline planners (Single-step / Multi-step MPC)

Used for perpendicular parking on clear scenes. Pipeline:

1. Geometric reference arc path (drive forward → reverse arc → straight in).
2. Kinematic bicycle model stepped under receding-horizon MPC tracking the
   reference with boundary penalties.
3. Full car-corner boundary checking at every step.

Limitations:

- Perpendicular parking only.
- No reasoning over arbitrary obstacles.
- Requires SciPy.

### Hybrid A* + Reeds-Shepp

Used for parallel parking, all obstacle scenarios, or when explicitly
selected. Search over `(x, y, θ)`:

- Forward and reverse motion primitives.
- Discrete steering set.
- Full car-rectangle collision checking against lane ∪ spot ∪ obstacles.
- **Reeds-Shepp analytic shot.** Every popped node within `rs_shot_radius` of
  the goal (and periodically when farther away) attempts a closed-form
  Reeds-Shepp curve directly to the goal. If the curve is collision-free and
  lands the car fully in the spot, it is accepted as the path tail and the
  search terminates with `used_analytic_shot = True`.
- **RS-based heuristic.** The A* heuristic combines Euclidean + heading with
  the Reeds-Shepp non-holonomic length, cached on a coarse `(Δx, Δy, Δθ)` grid.
- **Lightweight smoothing.** Near-duplicate and collinear waypoints are
  removed while preserving the first and final pose, rejecting any cleanup
  segment that collides.

### Q-learning (RL)

Discretizes the rear-axle state into `(ix, iy, ith)` buckets (0.45 m XY,
12 heading bins). 10 discrete `(gear × steer)` actions. ε-greedy training with
distance + heading shaping, collision penalty, and a large terminal bonus on
fully-in-spot success. Uses a **reverse-curriculum** schedule: early episodes
start near the goal, sampling radius and probability of using the true start
grow with training progress. Time-bounded so a single `plan_qlearn` call
returns in ≤ 30 s.

Honest limitation: tabular Q on continuous SE(2) with sparse rewards rarely
converges to a fully-in-spot greedy rollout in a few seconds of training. The
planner is included as a *learned baseline* — see the figures in §8 and the
discussion in `UPDATE.md`.

## 6. Pure Pursuit closed-loop tracker

When enabled, the tracker (`tracker.py`) splits the planned path into
single-gear segments and runs Pure Pursuit on each segment under the same
bicycle-model dynamics. Reverse segments flip both the lookahead frame and
the steering sign — without that flip the agent drives away from the path
(`θ̇ = v·tan(δ)/L` inverts with `v`).

Enable from the simulator with `AUTOPARK_TRACK=1`. Enable from the evaluator
with `--track`. The executed path is drawn in amber over the planned blue line;
toggle the overlay with `T`.

Tracker metrics:

| Metric | Meaning |
|---|---|
| `tracker_success` | Tracker reached the planned terminal pose AND the car is fully in spot |
| `mean_cte_m`, `max_cte_m` | Cross-track error (m) over the executed trajectory |
| `cusps` | Number of direction-reversal transitions in the planned path |
| `exec_final_pos_error_m` | Executed final position error vs the planner goal |
| `exec_final_heading_error_deg` | Executed final heading error |
| `exec_fully_in_spot` | All four executed corners inside the spot rectangle |

Typical accuracy on Hybrid A* paths at default dimensions: mean CTE 0.01–0.03 m,
executed final error 7–20 mm.

## 7. Planner Metrics

Every planner returns a uniform metrics dictionary:

| Metric | Meaning |
|---|---|
| `planning_time_s` | Wall time spent planning |
| `iterations` | Search-loop iterations (Hybrid A*) or training episodes (Q-learn) |
| `expanded_states` | Number of stored states (Hybrid A*) or Q-table size (Q-learn) |
| `path_length_m` | Geometric path length |
| `raw_path_length_m` / `smoothed_path_length_m` | Hybrid A* pre/post smoothing |
| `waypoints` / `raw_waypoints` / `smoothed_waypoints` | Path counts |
| `final_pos_error_m` | Final distance from goal pose |
| `final_heading_error_deg` | Final heading error |
| `fully_in_spot` | Final car rectangle fully inside the spot |
| `obstacles` | Number of active obstacle rectangles |
| `rs_shot_attempts` / `rs_shot_successes` / `used_analytic_shot` | Hybrid A*+RS |
| `training_time_s` / `successful_episodes` | Q-learning |

`evaluate.py` is the report-facing CLI. CLI arguments:

- `--mode perpendicular|parallel|all` (default `all`)
- `--scenario clear|obstacle|tight|none|entry_blocker|tight_lane|pillar_near_entry|parked_cars|all` (default `clear`)
- `--planner baseline|hybrid_astar|qlearn|all` (default `all`)
- `--sweep lane_width|spot_size|car_size|none` (default `none`)
- `--track` — also run the Pure Pursuit tracker and record tracking metrics
- `--output path.csv` — write CSV to file; otherwise CSV goes to stdout

When `--output` is omitted, CSV rows are written to stdout and the summary is
written to stderr. When `--output` is provided, the CSV is written to that
file and the summary is printed to stdout.

## 8. Report Figures

`plot_results.py` reads an evaluator CSV and writes PNGs into the chosen
output directory (default `results/figures/`). Generated figures:

- `<stem>_success_rate.png` — bar chart of success rate per planner.
- `<stem>_planning_time.png` — planning-time distribution (boxplot per planner).
- `<stem>_path_length.png` — mean path length (with std) per planner, on
  successful runs only.
- `<stem>_tracking_error.png` — mean / max cross-track error per planner
  (only when the CSV has tracker columns).
- `<stem>_sweep_<param>_<metric>.png` — for each `--sweep` value, mean line
  plus scatter cloud per planner vs the swept parameter.

Example workflow producing the figures committed to `results/figures/`:

```bash
python evaluate.py --mode all --scenario all --planner all --track \
    --output results/main.csv
python evaluate.py --mode all --scenario all --sweep lane_width \
    --planner hybrid_astar --output results/sweep_lane_width.csv
python evaluate.py --mode perpendicular --scenario all --sweep car_size \
    --planner all --output results/sweep_car_size.csv

python plot_results.py results/main.csv             --outdir results/figures
python plot_results.py results/sweep_lane_width.csv --outdir results/figures
python plot_results.py results/sweep_car_size.csv   --outdir results/figures
```

Summary of headline results from the default run (`results/main.csv`):

- **Success rate:** Hybrid A*+RS 90 %, baseline MPC 10 %, Q-learning 0 %.
- **Planning time:** baseline ~0.2 s, Hybrid A* ~1–7 s, Q-learning ~5–6 s
  (training-dominated, time-budget capped).
- **Tracking:** Pure Pursuit reproduces the planned path with mean CTE
  ≈ 0.02 m and executed final error ≈ 0.01 m on successful Hybrid A* paths.

## 9. Recommended Demo Flow

1. **Reverse into Spot** + **Clear** + **Single-step MPC** — see the baseline arc.
2. **Reverse into Spot** + **Clear** + **Hybrid A*** — note the smoother
   trajectory and the Reeds-Shepp shot counter in the HUD.
3. **Reverse into Spot** + **Parked Cars** + **Hybrid A*** — obstacle-aware
   planning around two parked cars.
4. **Parallel Parking** + **Clear** + **Hybrid A*** — multi-cusp parallel maneuver.
5. **Reverse into Spot** + **Clear** + **Q-learning (RL)** — observe the
   learned policy oscillating near the spot entry (expected limitation).
6. Re-run the simulator with `AUTOPARK_TRACK=1` to see the executed Pure
   Pursuit overlay in amber on top of the planned path.
7. Run `python evaluate.py --planner all --track --output results/main.csv`
   and then `python plot_results.py results/main.csv` to regenerate the
   report figures.

## 10. Current Limitations

- Tabular Q-learning does not converge to a fully-in-spot greedy rollout on
  the default dimensions within a 30 s training budget; a DQN extension or a
  hybrid Hybrid-A*-warmstart would be the natural next step.
- Reeds-Shepp module covers CSC + CCC (24-word family with time-flip /
  reflect symmetries); CCSC / CCSCC families are not yet included.
- Obstacles are simple static rectangles, not sensor-derived dynamic obstacles.
- Tracker speed and lookahead are fixed; an adaptive controller would handle
  high-curvature paths better.
- There is no CARLA integration yet.
- The baseline MPC planner depends on SciPy.

## 11. CARLA Stretch Goal

The same Hybrid A* + Reeds-Shepp planner and Pure Pursuit controller can be
driven against a live CARLA server. Three modules cooperate:

- `carla_bridge.py` — lazy-imports `carla`, owns the client/world lifecycle,
  converts CARLA world coordinates to our planner frame, and extracts nearby
  obstacles from `world.get_actors()` (vehicles, static props). A LiDAR-based
  occupancy helper is also available.
- `carla_controller.py` — Pure Pursuit controller that emits a
  `ControlCommand` (steer ∈ [-1, 1], throttle ∈ [0, 1], brake ∈ [0, 1],
  `reverse: bool`). A small `control_to_carla()` adapter wraps it into a real
  `carla.VehicleControl` only when CARLA is importable.
- `carla_demo.py` — end-to-end CLI:
  - `--probe` prints whether `carla` is importable.
  - `--dry-run` (default) runs the full pipeline against our internal
    bicycle integrator so the bridge wiring is verifiable without CARLA.
  - `--carla --host <h> --port <p> --town <map>` runs on a live server:
    spawn the ego near the spectator, extract obstacles in a 25 m radius,
    plan, then drive via `apply_control()` until the controller reports done
    or `--max_seconds` elapses.

### Install (workstation with CARLA)

```bash
# Install matching wheel for your CARLA server version, e.g.:
pip install carla
```

The `carla` PythonAPI typically requires Python 3.7–3.10. Our planner and
controller are pure Python ≥ 3.9; running the bridge needs the older Python
that matches your CARLA install.

### Dry-run verification

This sandbox does not have CARLA, so the dry-run path is what was tested:

```bash
python carla_demo.py --probe
# CARLA_AVAILABLE=False

python carla_demo.py --dry-run --mode perpendicular
# planner_ok=True planning_time=1.0s wp=20 | executed_ok=True
# final_err=0.02m heading_err=28deg mean_cte=0.40m

python carla_demo.py --dry-run --mode parallel
python carla_demo.py --dry-run --mode perpendicular --scenario parked_cars
```

The dry-run integrator includes throttle/brake/rolling-decel dynamics, so the
controller's tracking error is higher than the kinematic-only `tracker.py`
demo. This is expected — the dry-run exercises the actuator-style API
(`steer/throttle/brake/reverse`) that CARLA uses; the kinematic tracker is the
optimistic upper-bound case.

### Live run

```bash
python carla_demo.py --carla --host localhost --port 2000 --mode perpendicular
```

Before running, position the CARLA spectator on the desired parking spot
(its location + yaw are used as the planner-frame origin). The demo:

1. Connects to the server in synchronous mode (`fixed_delta_seconds=0.05`).
2. Spawns a Tesla Model 3 (`vehicle.tesla.model3`) at `spot_offset_xy` from
   the spectator.
3. Reads the ego's bounding box to fill `CarConfig` (length / width /
   wheelbase).
4. Calls `extract_static_obstacles(world, ..., radius_m=25, frame=...)` to
   pull nearby vehicles and static props into our `Rect[]` format.
5. Runs Hybrid A* + Reeds-Shepp on the obstacle set and the chosen mode.
6. Drives the planned path via `CarlaPurePursuitController` until the
   controller reports done.
7. Reports planning time, executed pose error, mean / max cross-track error.

### Known limitations

- The bridge assumes a roughly axis-aligned parking spot in the spectator
  frame. For arbitrary spot orientations, set `LocalFrame.yaw_offset_deg`
  before extracting obstacles.
- Obstacle bounding boxes are over-approximated as `max(extent.x, extent.y)`
  squares so yaw-rotated vehicles are conservatively covered. The planner's
  full car-rectangle collision check absorbs the slight over-conservatism.
- LiDAR-based occupancy (`occupancy_from_lidar`) is implemented but not
  wired into the default demo loop; it's intended for sensor-realistic
  experiments where you want to *not* rely on ground-truth actor poses.
- Controller gains in `ControlConfig` are tuned for slow parking; the live
  CARLA car may need wheelbase-specific re-tuning of throttle_kp, brake_kp,
  and lookahead.

## 12. Suggested Next Improvements

- Add CCSC / CCSCC Reeds-Shepp words for marginally shorter terminal arcs.
- Function-approximation RL (DQN / PPO) with a small CNN over the local
  occupancy patch, evaluated against the tabular baseline.
- Warm-start Q-learning from Hybrid A* solutions (DAgger-style imitation).
- Adaptive Pure Pursuit lookahead based on path curvature.
- Sensor-derived dynamic obstacles (CARLA LiDAR / depth).
- CARLA parking-lot scene with full sim-to-pygame planner handoff.
