# UPDATE Log

A running log of major implementation updates on the `mason-implement` branch.
For overall scope and roadmap, see `PROJECT_SCOPE.md`. For historical work that
predates this log, see `RECORD.md`.

---

## 2026-05-28 — Reeds-Shepp shot, Pure Pursuit tracker, Q-learning RL baseline, visualization, plots

This push fills out Stages 3–6 of `PROJECT_SCOPE.md` (algorithm comparison,
path-quality improvement, closed-loop tracking, visualization) and adds an
RL-based comparison baseline that the original scope listed as a stretch idea.

### Reeds-Shepp analytic shot for Hybrid A* (`reeds_shepp.py`, `hybrid_astar.py`)

- Added a self-contained Reeds-Shepp shortest-path generator covering the
  12-word CSC + CCC base set, expanded under the standard time-flip / reflect
  symmetry to cover the full 24-word reverse-aware family.
- Integrated as a Hybrid A* analytic shot:
  - At every popped node within `rs_shot_radius`, and periodically when
    farther away, the planner tries a closed-form RS curve from the current
    pose directly to the goal.
  - The RS sample list is validated against the same occupancy/lane/spot
    collision check used by motion-primitive expansion; only collision-free
    shots that land the car fully inside the spot are accepted.
  - When accepted, the RS curve is appended as the path tail and the search
    terminates with `metrics["used_analytic_shot"] = True`.
- Upgraded the Hybrid A* heuristic to combine Euclidean + heading with a
  Reeds-Shepp non-holonomic lower bound. RS heuristic values are cached on a
  coarse `(Δx, Δy, Δθ)` grid so each node costs amortised O(1).
- Diagnostic counters `rs_shot_attempts` and `rs_shot_successes` are populated
  on every plan call and surfaced in the metrics dict + evaluator CSV.

**Result on default-dimension scenes (CPython, no SciPy):**
- Perpendicular clear: planning ≈ 3s, path 17.7 m, 1 RS shot accepted.
- Parallel clear: planning ≈ 7.7s (down from ≈ 12s pre-cache), path 16.2 m.
- Path lengths consistently dropped because the RS final segment is
  geometrically optimal for the last few metres.

### Pure Pursuit closed-loop tracker (`tracker.py`, `trajectory.py`, `simulation.py`)

- Added a gear-aware Pure Pursuit tracker. The planned path is split into
  single-gear segments using the sign of the heading-projected motion; each
  segment is tracked under the kinematic bicycle model with the same wheelbase
  and steering limits as the planner.
- Reverse segments flip both the lookahead frame and the steering sign.
  Without the steering sign flip the agent drives away from the path because
  `θ̇ = v·tan(δ)/L` inverts with `v` — this was a real bug encountered during
  bring-up and is now called out in the code comment.
- Planned paths are densified to ≤ 0.15 m spacing before tracking so the
  smoother's collinear-point removal does not break gear inference.
- The tracker emits an executed `Waypoint` list and metrics:
  - `mean_cte_m`, `max_cte_m` (cross-track error),
  - `cusps` (number of direction reversals),
  - `exec_final_pos_error_m`, `exec_final_heading_error_deg`,
  - `exec_fully_in_spot`, `tracker_success`.
- Wired through `plan_trajectory(..., track=True)` so the simulator and
  evaluator can both opt into closed-loop tracking. `AUTOPARK_TRACK=1` enables
  it from the simulator.

**Tracking accuracy on default Hybrid A* paths:**
- Perpendicular: mean CTE 0.021 m, max 0.072 m, executed final error 7 mm.
- Parallel: mean CTE 0.009 m, max 0.036 m, executed final error 17 mm.

### Tabular Q-learning RL planner (`rl_qlearn.py`)

- Implemented as a *learned-policy comparison baseline* against the
  classical-control (MPC) and search-based (Hybrid A*) planners.
- State: discretised `(ix, iy, ith)` with 0.45 m XY resolution and 12 heading
  bins. Action: 10 discrete `(gear, steer)` combinations.
- Reward: per-step penalty + potential-based shaping on distance and heading,
  collision penalty, and a large terminal bonus when the car is fully inside
  the spot.
- **Reverse curriculum:** early episodes sample starts near the goal so the
  agent learns the terminal phase first; sampling radius and probability of
  using the true start both grow with training progress. This is the standard
  trick for sparse-reward navigation.
- ε-greedy training with exponential decay, time-bounded so a `plan_qlearn`
  call returns in ≤ 30 s.
- Reports `training_time_s`, `episodes`, `successful_episodes`,
  `expanded_states` (Q-table size), plus the standard pose / path metrics on
  the greedy rollout.

**Observed result on the default scenes:** the agent learns useful Q values
near the goal (curriculum successes 100s out of 40k episodes) but the greedy
rollout typically gets stuck in oscillation near the spot entry and fails to
fully park. This is the expected limitation of pure tabular RL on continuous
SE(2) with sparse rewards, and is itself a useful comparison point in the
final report: it motivates either function approximation (DQN), or hybrid
approaches that warm-start the policy from Hybrid A* solutions.

### Visualization upgrades (`simulation.py`)

- Window enlarged to 1080×720 so the HUD does not overflow when planner-
  specific metric lines are shown.
- New keys:
  - `G` — toggle occupancy-grid overlay (free cells outlined faintly; blocked
    cells in red).
  - `T` — toggle the executed (closed-loop) trajectory overlay (drawn in amber
    over the planned blue line and the green travelled-portion line).
- HUD now shows Reeds-Shepp shot counters when applicable, RL training stats
  when the planner is Q-learning, and tracking-error metrics when the tracker
  is enabled.

### Evaluator + report plot script (`evaluate.py`, `plot_results.py`)

- Evaluator gained `qlearn` as a planner choice (`--planner qlearn` or
  `--planner all` includes it), and a `--track` flag that runs the Pure
  Pursuit tracker on each plan and records the tracking metrics.
- CSV schema extended with `rs_shot_*`, `used_analytic_shot`, tracker
  metrics, and RL training stats. Existing columns unchanged.
- Added `plot_results.py` which consumes an evaluator CSV and writes
  matplotlib figures to `results/figures/`:
  - success rate per planner,
  - planning-time distribution per planner,
  - mean path length per planner with error bars,
  - per-sweep line plots for planning time, path length, and final error,
  - tracking-error boxplots when `--track` was set.

### Settings UI (`settings_window.py`, `config.py`)

- Added a fourth planner radio button "Q-learning (RL)".
- `ParkingConfig.planner` now accepts `"qlearn"` and is dispatched through
  `plan_trajectory`.

### Docs

- `README.md` rewritten so the three-way planner comparison (classical MPC /
  search-based Hybrid A* / learned Q-learning), Pure Pursuit tracking, the
  Reeds-Shepp shot, and the new HUD/key bindings are all described.
- `USER_GUIDE.md` will be refreshed in the next pass.

### Validation

- `python3 -m py_compile` on every module: clean.
- `python3 evaluate.py --mode perpendicular --scenario clear --planner all --track --output /tmp/autopark_smoke.csv`:
  3 rows produced (baseline cleanly skipped — no SciPy locally; Hybrid A*
  succeeded with `used_analytic_shot=True` and tracker reached spot with
  ~7 cm executed error; Q-learn cleanly reported its rollout failure).

### Known limitations / next steps

- Tabular Q-learning does not converge to a fully-in-spot rollout on the
  default dimensions within a 30 s training budget. A DQN extension, or
  warm-starting Q values from Hybrid A* solutions, would be the natural next
  step but is out of scope for this push.
- The Reeds-Shepp module covers CSC + CCC (24-word family). Adding the
  CCSC/CCSCC families would marginally improve path optimality but is rarely
  needed for parking-sized maneuvers.
- The plot script imports matplotlib lazily and prints a clean install
  message if it is not present; CI/runners that need plots should
  `pip install matplotlib`.
