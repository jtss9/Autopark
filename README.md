# Smart Parking Simulator

A top-down 2D parking simulation built for the *Introduction to Smart Cars* course final project.  
Configure road and vehicle dimensions, then watch the car automatically execute a parking maneuver guided by an MPC controller.

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
- **Parallel Parking (路邊停車)** — not yet implemented

If the car is too large for the spot, the spot outline turns red and simulation is blocked until dimensions are corrected.

Click **Start Simulation** to proceed.

### Phase 2 — Simulation Window

Animates the full parking trajectory computed by the MPC planner.

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `R` | Restart animation |
| `S` | Return to settings (preserves last slider values) |
| `ESC` / `Q` | Quit |

The HUD shows parking type, dimensions, current phase name, and step counter.

**Feasibility**: if the car body would clip a boundary during the maneuver, the animation plays up to the collision frame, then freezes — the car turns red and a **COLLISION** overlay is displayed.

---

## File Structure

```
final/
├── main.py              # Entry point: settings → simulation loop
├── config.py            # CarConfig and ParkingConfig dataclasses
├── parking_lot.py       # World-space geometry (lane, spot, car corners)
├── settings_window.py   # Phase 1: tkinter settings UI with live preview
├── trajectory.py        # Trajectory planner (geometric reference + MPC)
├── controller.py        # Bicycle kinematic model and MPC controller
├── simulation.py        # Phase 2: pygame animation loop
└── requirements.txt
```

---

## Coordinate System

All geometry uses a standard math convention: **+x right, +y up**, units in metres.  
Both the tkinter canvas and pygame window flip the y-axis for rendering (`screen_y = origin_y − world_y × scale`).

The reference point for all waypoints and car corner calculations is the **rear axle centre**.

---

## Trajectory Planning

Parking uses a two-stage pipeline:

1. **Geometric reference** — a 3-phase arc path (drive forward → reverse arc → straight into spot) computed analytically from the minimum turning radius.
2. **MPC simulation** — a kinematic bicycle model is stepped forward under a receding-horizon MPC (horizon N=5, dt=0.05 s) that tracks the reference while penalising boundary violations. All four car body corners are checked at every step; any violation terminates planning and returns the collision waypoints for animation.

```
wheelbase       = car_length × 0.65
max_steer_angle = 35°
R_min           = wheelbase / tan(max_steer_angle)  ≈ 3–5 m
```
