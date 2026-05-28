# Smart Parking Simulator

A top-down 2D parking simulation built for the *Introduction to Smart Cars*
course final project. Configure road and vehicle dimensions, pick a parking
mode (perpendicular / parallel), a scenario (clear / entry-blocker / pillar /
parked-cars / tight-lane) and a planner (geometric MPC baseline / Hybrid A* with
Reeds-Shepp analytic shot / tabular Q-learning), then watch the car execute
the maneuver. Optionally turn on the Pure Pursuit closed-loop tracker to see
the executed (control-tracked) trajectory laid against the planned one.

The project is framed as a map-based autonomous parking stack; see
`PROJECT_SCOPE.md` for the full target architecture. For day-to-day operation,
see `USER_GUIDE.md`. Major in-progress work and design decisions are recorded
in `UPDATE.md`.

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
python evaluate.py                                  # default sweep
python evaluate.py --mode all --scenario all --planner all --output results/all.csv
python evaluate.py --planner hybrid_astar --track --output results/tracked.csv
python plot_results.py results/all.csv              # writes PNGs to results/figures/
```

---

## Usage

### Phase 1 — Settings Window

Adjust all parameters using the sliders on the left. The preview canvas on the right updates in real time.

| Slider | Range |
|---|---|
| Lane Width | 3.5 – 5.5 m |
| Spot Length | 5.0 – 6.0 m |
| Spot Width | 2.0 – 3.0 m |
| Car Length | 3.5 – 5.0 m |
| Car Width | 1.6 – 2.2 m |

**Parking Type**
- **Reverse into Spot (倒車入庫)** — car starts parallel to the road and reverses perpendicularly into the spot
- **Parallel Parking (路邊停車)** — planned with Hybrid A*

**Scenario**
- **Clear** — empty parking environment
- **Entry Blocker** — adds one occupied region near the parking entry
- **Tight Lane** — used by the evaluator as a narrow-lane scenario
- **Parked Cars** — adds parked-car obstacle rectangles around the target

**Planner**
- **Single-step MPC** — tracks a pre-computed geometric arc; succeeds in wider lanes, fails with COLLISION in narrow ones
- **Multi-step MPC** — uses alternating reverse/correction attempts; extends the feasibility boundary into narrower lanes where single-step fails
- **Hybrid A*** — map-based planner with Reeds-Shepp analytic shot for perpendicular, parallel, and obstacle-aware scenarios
- **Q-learning (RL)** — tabular Q-learning agent with reverse-curriculum training over a discretized (x, y, θ) grid; included as a learned-policy comparison baseline

If the car is too large for the spot, the spot outline turns red and simulation is blocked until dimensions are corrected.

Click **Start Simulation** to proceed.

### Phase 2 — Simulation Window

Animates the full parking trajectory computed by the selected planner.

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `R` | Restart animation |
| `G` | Toggle occupancy-grid overlay |
| `T` | Toggle executed (closed-loop) path overlay |
| `S` | Return to settings (preserves last slider values) |
| `ESC` / `Q` | Quit |

The HUD shows parking type, active planner, scenario, dimensions, status,
current phase name, step counter, final containment, and planner metrics when
available.

**Feasibility**: if the car body would clip a boundary or the final pose is not
fully inside the spot, the animation plays to the last available waypoint, the
car turns red, and a **FAILED** overlay plus concrete failure message is shown.

---

## File Structure

```
final/
├── main.py              # Entry point: settings → simulation loop
├── config.py            # CarConfig and ParkingConfig dataclasses
├── parking_lot.py       # World-space geometry (lane, spot, car corners)
├── scenarios.py         # Static obstacle scenarios
├── settings_window.py   # Phase 1: tkinter settings UI with live preview
├── trajectory.py        # Trajectory planner dispatch + result/tracker glue
├── hybrid_astar.py      # Hybrid A* planner over (x, y, θ) with Reeds-Shepp shot
├── reeds_shepp.py       # Reeds-Shepp shortest-path generator (CSC + CCC, full RS via symmetries)
├── tracker.py           # Pure Pursuit closed-loop tracker with gear-aware cusps
├── rl_qlearn.py         # Tabular Q-learning planner with reverse curriculum
├── controller.py        # Bicycle kinematic model and MPC controller
├── simulation.py        # Phase 2: pygame animation loop
├── evaluate.py          # Batch CSV evaluation runner with --track and --planner all
├── plot_results.py      # Render report figures from evaluator CSV
├── PROJECT_SCOPE.md     # Upgraded final-project scope and implementation plan
├── USER_GUIDE.md        # How to run and interact with the simulator
├── UPDATE.md            # Running log of major implementation updates
└── requirements.txt
```

---

## Coordinate System

All geometry uses a standard math convention: **+x right, +y up**, units in metres.  
Both the tkinter canvas and pygame window flip the y-axis for rendering (`screen_y = origin_y − world_y × scale`).

The reference point for all waypoints and car corner calculations is the **rear axle centre**.

---

## Trajectory Planning

### Single-step MPC (default planner)

Uses a two-stage pipeline:

1. **Geometric reference** — a 3-phase arc path (drive forward → reverse arc → straight into spot) computed analytically from the minimum turning radius.
2. **MPC simulation** — a kinematic bicycle model is stepped forward under a receding-horizon MPC (horizon N=5, dt=0.05 s) that tracks the reference while penalising boundary violations. All four car body corners are checked at every step; any violation terminates planning and returns the collision waypoints for animation.

### Multi-step MPC

Extends feasibility into narrow lanes where single-step fails. Algorithm:

1. **Drive forward** to the initial arc start position.
2. **Reverse (attempt N)** — fresh arc reference rebuilt from the car's current heading; MPC tracks it toward the spot.
3. **Correct** if any corner approaches the road edge: Phase A reverses with full right steer to regain y-clearance; Phase B drives forward to a safe x-position.
4. Repeat 2–3 up to 5 times; declare success when the rear axle reaches the spot centre.

**Demo case** (feasibility improvement):
Set Car Length = 3.8 m, Car Width = 1.6 m, Lane Width = 4.2 m.  
Single-step → COLLISION. Multi-step → 2-attempt success.

```
wheelbase       = car_length × 0.65
max_steer_angle = 35°
R_min           = wheelbase / tan(max_steer_angle)  ≈ 3–5 m
```

### Hybrid A* with Reeds-Shepp analytic shot

The Hybrid A* planner (`hybrid_astar.py`) searches over `(x, y, θ)` using
forward/reverse bicycle-model motion primitives and validates every candidate
pose with full car-corner collision checking against the lane, spot, and
obstacle rectangles. Three upgrades over a plain Hybrid A*:

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

### Pure Pursuit closed-loop tracker

`tracker.py` splits the planned path into single-gear segments (cusps detected
by the sign of the heading-projected motion) and runs a standard Pure Pursuit
controller on each segment under the same kinematic bicycle model. The reverse
segments flip both the lookahead frame and the steering sign — without that
sign flip the agent drives away from the path because
`θ̇ = v·tan(δ)/L` inverts with `v`. Output metrics include cusp count, mean
and max cross-track error, executed-final position/heading error, and whether
the executed (control-tracked) pose ends fully inside the spot.

### Tabular Q-learning RL planner

`rl_qlearn.py` discretizes the rear-axle state into `(ix, iy, ith)` buckets,
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

`evaluate.py` writes a stable CSV with every column above and is the
report-facing CLI:

```bash
python evaluate.py
python evaluate.py --mode parallel --scenario all --planner hybrid_astar
python evaluate.py --mode all --scenario all --planner all --output results/all.csv
python evaluate.py --mode all --scenario all --sweep lane_width --planner all --output results/lane_width.csv
python evaluate.py --planner hybrid_astar --track --output results/tracked.csv
python plot_results.py results/all.csv
```
