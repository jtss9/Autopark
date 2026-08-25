# Smart Parking Simulator

A top-down 2D parking simulation built for the *Introduction to Intelligent Vehicles*
course final project. Configure road and vehicle dimensions, pick a parking
mode (perpendicular / parallel) and a planner (geometric MPC / Hybrid A* with
Reeds-Shepp analytic shot / tabular Q-learning), then watch the car execute
the maneuver. With Hybrid A* you can drop a draggable obstacle in the preview
and the planner routes around it. Optionally turn on the Pure Pursuit
closed-loop tracker to see the executed (control-tracked) trajectory laid
against the planned one.

---

## Project layout

`main.py` is the only Python file in the repository root; all other source
modules live in `src/`. `main.py` adds `src/` to the import path at startup, so
the modules keep flat imports. Run the app from the root with `python main.py`;
run the helper scripts as `python src/<script>.py`.

---

## Requirements

- Python 3.11
- pygame >= 2.0
- scipy >= 1.7
- Pillow >= 9.0

Install dependencies:

```bash
conda activate car
pip install -r requirements.txt
```

---

## How to Run

```bash
conda activate car
cd path/to/final
python main.py
```

To pick a planner via environment variable (overrides the Settings UI choice):

```bash
AUTOPARK_PLANNER=hybrid_astar python main.py
AUTOPARK_PLANNER=qlearn python main.py
```

To overlay the Pure Pursuit closed-loop tracker:

```bash
AUTOPARK_TRACK=1 AUTOPARK_PLANNER=hybrid_astar python main.py
```

To run batch evaluation metrics:

```bash
python src/evaluate.py                                  # default sweep
python src/evaluate.py --mode all --scenario all --planner all --output results/all.csv
python src/evaluate.py --planner hybrid_astar --track --output results/tracked.csv
python src/plot_results.py results/all.csv              # writes PNGs to results/figures/
```

---

## Usage

### Phase 1 — Settings Window

Adjust all parameters using the sliders on the left. The preview canvas on the
right updates in real time and annotates all five dimensions (lane width, spot
length/width, car length/width).

| Slider | Range |
|---|---|
| Lane Width | 3.5 – 5.5 m |
| Spot Length | 5.0 – 6.0 m |
| Spot Width | 2.0 – 3.0 m |
| Car Length | 3.5 – 5.0 m |
| Car Width | 1.6 – 2.2 m |

**Parking Type**
- **Reverse into Spot (倒車入庫)** — car starts parallel to the road and reverses perpendicularly into the spot (single-step MPC)
- **Parallel Parking (路邊停車)** — two-arc S-curve tracked by MPC into the curb-side spot

**Planner**
- **Single-step MPC** — tracks a pre-computed geometric arc; succeeds in wider lanes, fails with COLLISION in narrow ones
- **Hybrid A*** — map-based planner with Reeds-Shepp analytic shot for perpendicular, parallel, and obstacle-aware scenarios
- **Q-learning (RL)** — tabular Q-learning agent with reverse-curriculum training over a discretized (x, y, θ) grid; included as a learned-policy comparison baseline

(The Multi-step MPC planner still exists in the code and is reachable via
`AUTOPARK_PLANNER=multi` / the evaluator, but is hidden in the UI because it
does not generalise cleanly across vehicle/spot sizes.)

**Obstacle (Hybrid A* only)**
When the planner is Hybrid A*, an **Add obstacle** checkbox appears. Tick it to
place a fixed-size (0.9 m) obstacle in the preview, then drag it into position.
The obstacle cannot be dropped on the car (start) or inside the spot (goal).
The same obstacle is drawn in the simulation and the Hybrid A* planner routes
around it.

If the car is too large for the spot, the spot outline turns red and simulation
is blocked until dimensions are corrected.

Click **Start Simulation** to proceed.

### Phase 2 — Simulation Window

Animates the full parking trajectory computed by the selected planner.

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `↑` / `↓` | Animation speed |
| `R` | Restart animation |
| `G` | Toggle occupancy-grid overlay |
| `T` | Toggle executed (closed-loop) path overlay |
| `S` | Return to settings (preserves last slider values) |
| `ESC` / `Q` | Quit |

The HUD shows parking type, active planner, dimensions, status, current phase
name, step counter, final containment, and planner metrics when available.

**Feasibility**: if the car body would clip a boundary or the final pose is not
fully inside the spot, the animation plays to the last available waypoint, the
car turns red, and a **FAILED** overlay plus concrete failure message is shown.

---

## File Structure

```
final/
├── main.py              # Entry point (only root .py): settings → simulation loop, bootstraps src/ on sys.path
├── requirements.txt
├── README.md · CLAUDE.md · SUMMARY.md
└── src/
    ├── config.py            # CarConfig and ParkingConfig dataclasses (incl. obstacle field)
    ├── parking_lot.py       # World-space geometry (lane, spot, car corners), Rect
    ├── scenarios.py         # Named obstacle scenarios (evaluator)
    ├── settings_window.py   # Phase 1: tkinter settings UI, live preview, draggable obstacle
    ├── trajectory.py        # Planner dispatch + MPC planners (perp/parallel) + early-stop + tracker glue
    ├── hybrid_astar.py      # Hybrid A* over (x, y, θ) with Reeds-Shepp shot; OccupancyGrid + SAT collision
    ├── reeds_shepp.py       # Reeds-Shepp shortest-path generator (CSC + CCC, full RS via symmetries)
    ├── tracker.py           # Pure Pursuit closed-loop tracker with gear-aware cusps
    ├── rl_qlearn.py         # Tabular Q-learning planner with reverse curriculum
    ├── controller.py        # Bicycle kinematic model and MPC controller
    ├── simulation.py        # Phase 2: pygame animation loop
    ├── evaluate.py          # Batch CSV evaluation runner with --track and --planner all
    ├── plot_results.py      # Render report figures from evaluator CSV
    ├── carla_bridge.py      # CARLA stretch: lazy import + connection + obstacle/pose helpers
    ├── carla_controller.py  # CARLA stretch: Pure Pursuit → carla.VehicleControl adapter
    └── carla_demo.py        # CARLA stretch: end-to-end demo (--carla or --dry-run)
```

---

## Coordinate System

All geometry uses a standard math convention: **+x right, +y up**, units in metres.  
Both the tkinter canvas and pygame window flip the y-axis for rendering (`screen_y = origin_y − world_y × scale`).

The reference point for all waypoints and car corner calculations is the **rear axle centre**.

---

## Trajectory Planning

### Geometric reference + MPC tracking

The default planner for both parking types is a single two-stage pipeline; only
the **geometry strategy** in stage 1 differs between perpendicular and parallel.

1. **Geometric reference** — an analytic reference path computed from the
   minimum turning radius. The shape depends on the parking type (below).
2. **MPC simulation** — a kinematic bicycle model is stepped forward under a
   receding-horizon MPC (horizon N=5, dt=0.05 s) that tracks the reference while
   penalising boundary violations. All four car body corners are checked at
   every step; any violation terminates planning and returns the collision
   waypoints for animation.
3. **Termination — early-stop ("park then align")** — instead of driving the
   full reference path (which would keep reversing and clip the spot's far edge),
   the loop stops the moment the whole car body is inside the spot **and** the
   heading is within `ALIGN_TOL = 8°` of the goal. It remembers the most-aligned
   in-spot pose; if a later step would leave bounds, it truncates back to that
   pose and reports success instead of a collision.

The geometry strategy in stage 1 is the only part that differs by parking type:

**Perpendicular — single-step (倒車入庫, `_plan_perpendicular` + `_plan_perpendicular_mpc`)**
A 3-phase arc: drive forward → 90° reverse arc → straight reverse into the spot.
Succeeds in wider lanes; fails with COLLISION in narrow ones.

**Perpendicular — multi-step (`_plan_perpendicular_multistep`) — ⚠ future work**
Aims to extend feasibility into narrow lanes by running alternating reverse /
forward-correction attempts (up to 5), rebuilding a fresh arc each time. Does
**not** generalise reliably across vehicle/spot sizes yet, so it is **hidden
from the Settings UI** and left as future work; still reachable via
`AUTOPARK_PLANNER=multi` and the evaluator for experimentation.

**Parallel — single-step (路邊停車, `_plan_parallel` + `_plan_parallel_mpc`)**
An equal-radius **two-arc S-curve**: drive forward to `x_stop`, reverse with
right steer through `α = arccos(1 − Δy/2R)` (where `Δy = lane_width/2 + spot_width/2`),
then reverse with left steer back to heading 0 inside the curb-side spot.

### Hybrid A* with Reeds-Shepp analytic shot

The Hybrid A* planner (`src/hybrid_astar.py`) searches over `(x, y, θ)` using
forward/reverse bicycle-model motion primitives and validates every candidate
pose against the lane, spot, and obstacle rectangles. Collision checking uses
the car-corner containment test **plus an exact Separating-Axis-Theorem (SAT)**
car-body-vs-obstacle intersection, so a small obstacle sitting under the middle
of a car edge (no corner touching it) is still caught. Three upgrades over a
plain Hybrid A*:

- **Reeds-Shepp analytic shot.** At every popped node within a configurable
  radius (and periodically when farther away), the planner tries a closed-form
  Reeds-Shepp curve from the current pose directly to the goal. If the curve
  is collision-free and lands the car fully inside the spot, it is accepted as
  the path tail and the search terminates. This is the classical Dolgov et al.
  (2008) trick that dramatically reduces search effort and produces smooth,
  vehicle-feasible terminal maneuvers.
- **RS-based non-holonomic heuristic.** Within the same radius, the heuristic
  combines the standard Euclidean + heading term with the Reeds-Shepp path
  length, cached on a coarse `(Δx, Δy, Δθ)` grid for amortised O(1) lookup.
- **Lightweight smoothing.** Near-duplicate and nearly collinear waypoints are
  removed while preserving the first and final pose and rejecting any cleanup
  segment that collides or leaves valid bounds.

Obstacles come from two sources: the interactive draggable obstacle
(`ParkingConfig.obstacle`) and the named scenarios used by the evaluator
(`scenarios.obstacles_for`). Both are merged in `plan_hybrid_astar`.

### Pure Pursuit closed-loop tracker

`src/tracker.py` splits the planned path into single-gear segments (cusps
detected by the sign of the heading-projected motion) and runs a standard Pure
Pursuit controller on each segment under the same kinematic bicycle model. The
reverse segments flip both the lookahead frame and the steering sign — without
that sign flip the agent drives away from the path because
`θ̇ = v·tan(δ)/L` inverts with `v`. Output metrics include cusp count, mean
and max cross-track error, executed-final position/heading error, and whether
the executed (control-tracked) pose ends fully inside the spot.

### Tabular Q-learning RL planner

`src/rl_qlearn.py` discretizes the rear-axle state into `(ix, iy, ith)` buckets,
exposes 10 discrete `(gear × steering)` actions, and trains a tabular Q
function with ε-greedy exploration, potential-based shaping on distance and
heading, and a large terminal bonus when the car is fully inside the spot.
A **reverse-curriculum** schedule samples episode starts near the goal early
in training and expands outward as training progresses. The greedy rollout is
returned as the "planned" path. It is intentionally included as a *learned
baseline*; on this continuous SE(2) problem with sparse rewards, tabular Q
typically fails to converge in a few seconds of training, which is itself a
useful comparison point against the search-based and classical-control
methods.

### Metrics

Each planner returns a uniform metrics dictionary: `planning_time_s`,
`iterations`, `expanded_states`, `path_length_m` (raw + smoothed where
applicable), `waypoints`, `final_pos_error_m`, `final_heading_error_deg`,
`fully_in_spot`, `obstacles`. Hybrid A* additionally reports
`rs_shot_attempts/successes` and `used_analytic_shot`. Q-learning adds
`training_time_s`, `successful_episodes`, `episodes`. When the tracker is
enabled the result also carries `mean_cte_m`, `max_cte_m`, `cusps`,
`exec_final_pos_error_m`, `exec_fully_in_spot`.

`src/evaluate.py` writes a stable CSV with every column above and is the
report-facing CLI:

```bash
python src/evaluate.py
python src/evaluate.py --mode parallel --scenario all --planner hybrid_astar
python src/evaluate.py --mode all --scenario all --planner all --output results/all.csv
python src/evaluate.py --mode all --scenario all --sweep lane_width --planner all --output results/lane_width.csv
python src/evaluate.py --planner hybrid_astar --track --output results/tracked.csv
python src/plot_results.py results/all.csv
```

### CARLA stretch goal

`src/carla_bridge.py`, `src/carla_controller.py`, and `src/carla_demo.py` wire
the same Hybrid A* + Reeds-Shepp planner and Pure Pursuit controller into a live
CARLA server. The architecture is:

- `carla_bridge.py` lazy-imports `carla`, owns the client/world lifecycle, and
  converts CARLA world coordinates to our planner frame. It exposes helpers
  for actor → obstacle Rect extraction and (optionally) LiDAR → occupancy
  voxel extraction.
- `carla_controller.py` is a Pure Pursuit controller whose output is a
  `ControlCommand` dataclass (steer, throttle, brake, reverse). A tiny
  adapter wraps it into a real `carla.VehicleControl` when needed.
- `carla_demo.py` ties the pipeline together:
  - `--dry-run` (default) runs the entire pipeline against our internal
    bicycle integrator — verifies wiring without needing CARLA.
  - `--carla` connects to a live server (`--host`, `--port`, `--town`),
    spawns the ego, extracts obstacles in a 25 m radius, plans, and drives.
  - `--probe` reports whether `carla` is importable.

Install CARLA (matching your Python version):

```bash
pip install carla        # version must match your CARLA server
```

Run end-to-end against a live server:

```bash
python src/carla_demo.py --carla --host localhost --port 2000 --mode perpendicular
```

Or run the dry-run pipeline (no CARLA needed):

```bash
python src/carla_demo.py --dry-run --mode parallel
python src/carla_demo.py --dry-run --mode perpendicular --scenario parked_cars
```
