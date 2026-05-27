# Smart Parking Simulator

A top-down 2D parking simulation built for the *Introduction to Smart Cars* course final project.  
Configure road and vehicle dimensions, then watch the car automatically execute a parking maneuver guided by an MPC controller.

The current MPC arc planner is the baseline. The project is being extended
toward a map-based autonomous parking stack with Hybrid A* planning; see
`PROJECT_SCOPE.md`.

For operating instructions, see `USER_GUIDE.md`.

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

To try the initial Hybrid A* planner in the simulator:

```bash
AUTOPARK_PLANNER=hybrid_astar python main.py
```

To run batch evaluation metrics:

```bash
python evaluate.py
python evaluate.py --mode all --scenario all --planner hybrid_astar --output results/demo.csv
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
- **Hybrid A*** — map-based planner for perpendicular, parallel, and obstacle-aware scenarios

If the car is too large for the spot, the spot outline turns red and simulation is blocked until dimensions are corrected.

Click **Start Simulation** to proceed.

### Phase 2 — Simulation Window

Animates the full parking trajectory computed by the selected planner.

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `R` | Restart animation |
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
├── trajectory.py        # Trajectory planner (geometric reference + MPC)
├── hybrid_astar.py      # Initial occupancy-grid Hybrid A* planner
├── controller.py        # Bicycle kinematic model and MPC controller
├── simulation.py        # Phase 2: pygame animation loop
├── evaluate.py          # Batch CSV evaluation runner
├── PROJECT_SCOPE.md     # Upgraded final-project scope and implementation plan
├── USER_GUIDE.md        # How to run and interact with the simulator
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

The new Hybrid A* path is opt-in with `AUTOPARK_PLANNER=hybrid_astar`. It
searches over `(x, y, theta)` using forward/reverse bicycle-model motion
primitives and validates every candidate pose with full car-corner collision
checking.

Parallel parking now uses Hybrid A* by default because the old geometric/MPC
baseline only supports perpendicular parking.

Hybrid A* reports metrics including planning time, expanded states, raw and
smoothed path length, waypoint count, final position/heading error, obstacle
count, and whether the final vehicle rectangle is fully inside the parking spot.

`evaluate.py` is the report-facing CLI. Useful examples:

```bash
python evaluate.py
python evaluate.py --mode parallel --scenario all --planner hybrid_astar
python evaluate.py --mode all --scenario all --sweep lane_width --planner hybrid_astar --output results/lane_width.csv
```
