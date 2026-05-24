# Smart Parking Simulator

A top-down 2D parking simulation built for the *Introduction to Smart Cars* course final project.  
The program lets you configure road and vehicle dimensions, then visualizes the initial parking scene.

---

## Requirements

- Python 3.11
- pygame >= 2.0
- Pillow >= 9.0 (installed but currently unused — reserved for future use)

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
| Lane Width | 3.0 – 5.0 m |
| Spot Length | 5.0 – 6.0 m |
| Spot Width | 2.0 – 3.0 m |
| Car Length | 3.5 – 5.0 m |
| Car Width | 1.6 – 2.2 m |

**Parking Type**
- **Reverse into Spot (倒車入庫)** — car starts parallel to the road and reverses perpendicularly into the spot
- **Parallel Parking (路邊停車)** — car starts parallel to the road and reverses into a roadside spot

If the car is too large for the spot, the spot outline turns red and simulation is blocked until dimensions are corrected.

Click **Start Simulation** to proceed.

### Phase 2 — Simulation Window

Displays a top-down view of the parking scene with the car at its starting position.

| Key | Action |
|---|---|
| `ESC` / `Q` | Quit |

---

## File Structure

```
final/
├── main.py              # Entry point
├── config.py            # CarConfig and ParkingConfig dataclasses
├── parking_lot.py       # World-space geometry (lane, spot, car start pose)
├── settings_window.py   # Phase 1: tkinter settings UI with live preview
├── simulation.py        # Phase 2: pygame static scene
├── car.png              # Car image asset (top-down view)
└── requirements.txt
```

---

## Coordinate System

All geometry uses a standard math convention: **+x right, +y up**, units in metres.  
Both the tkinter canvas and pygame window flip the y-axis for rendering (`screen_y = origin_y − world_y × scale`).

---

## Parking Geometry

The lane length is fixed regardless of slider values, so only the car and spot change size as you adjust the sliders.

- **Perpendicular** lane length: 23.0 m
- **Parallel** lane length: 23.5 m

The minimum turning radius is derived from the car's wheelbase and maximum steering angle:

```
wheelbase       = car_length × 0.65
max_steer_angle = 35°
R_min           = wheelbase / tan(max_steer_angle)  ≈ 3–5 m
```
