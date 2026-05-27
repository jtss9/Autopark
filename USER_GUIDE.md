# User Guide

## 1. Purpose

This simulator demonstrates autonomous parking in a structured parking-lot scene.
It supports:

- Reverse perpendicular parking.
- Parallel parking.
- Clear and obstacle scenarios.
- Baseline geometric/MPC parking for simple perpendicular cases.
- Hybrid A* parking for map-based, obstacle-aware planning.
- Planner metrics for final-project reporting.

## 2. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python main.py
```

Run the app with Hybrid A* explicitly enabled:

```bash
AUTOPARK_PLANNER=hybrid_astar python main.py
```

Run batch evaluation:

```bash
python evaluate.py
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

The preview updates immediately when a slider changes.

If the car cannot fit inside the selected parking spot, the spot outline turns
red and the simulation cannot start.

### Parking Type

Choose one:

- **Reverse into Spot**: perpendicular reverse parking.
- **Parallel Parking**: curb-side parallel parking.

### Scenario

Choose one:

- **Clear**: no extra obstacle.
- **Obstacle**: adds one occupied region to the parking map.

Obstacle scenarios automatically use Hybrid A* because the baseline planner does
not reason over occupancy-grid obstacles.

## 4. Simulation Window

After clicking **Start Simulation**, the pygame simulation window opens.

### Controls

| Key | Action |
|---|---|
| `SPACE` | Pause or resume animation |
| `R` | Restart current animation |
| `S` | Return to settings window |
| `ESC` / `Q` | Quit |

### HUD Fields

The HUD shows:

- Parking type.
- Active planner.
- Scenario.
- Lane, spot, and car dimensions.
- Minimum turning radius.
- Current phase.
- Current step.
- Planner metrics when available.

### What Step Means

`Step` is the current waypoint index in the planned trajectory.

Example:

```text
Step: 20 / 36
```

This means the animation is currently showing waypoint 20 out of 36. It is not
seconds or real time.

Parallel parking is animated slower, advancing one waypoint per frame.

## 5. Planners

### Baseline Planner

The baseline planner is used for perpendicular parking when no obstacle scenario
is enabled.

It uses:

1. A geometric reference path.
2. A short-horizon MPC controller.
3. Full car-corner boundary checks.

Limitations:

- Only supports perpendicular parking.
- Does not plan around arbitrary obstacles.
- Requires SciPy.

### Hybrid A* Planner

Hybrid A* is used when:

- `AUTOPARK_PLANNER=hybrid_astar` is set.
- Parallel parking is selected.
- An obstacle scenario is selected.

It searches over:

```text
x, y, theta
```

The planner uses:

- Forward and reverse motion primitives.
- Discrete steering actions.
- Bicycle-model vehicle motion.
- Occupancy-grid obstacle checks.
- Full car-rectangle collision checking.
- Strict success checking that requires the car to be fully inside the parking spot.

## 6. Planner Metrics

Hybrid A* reports metrics that are useful for final-project analysis:

| Metric | Meaning |
|---|---|
| `planning_time_s` | Time spent planning |
| `iterations` | Search-loop iterations |
| `expanded_states` | Number of stored search states |
| `path_length_m` | Total geometric path length |
| `waypoints` | Number of output waypoints |
| `final_pos_error_m` | Final distance from goal pose |
| `final_heading_error_deg` | Final heading error in degrees |
| `fully_in_spot` | Whether the final car rectangle is fully inside the spot |
| `obstacles` | Number of active obstacle rectangles |

Use:

```bash
python evaluate.py
```

This prints CSV rows for:

- Perpendicular clear.
- Perpendicular obstacle.
- Parallel clear.
- Parallel obstacle.

## 7. Recommended Demo Flow

1. Start with **Reverse into Spot** and **Clear**.
2. Run the baseline behavior.
3. Run with Hybrid A*:

   ```bash
   AUTOPARK_PLANNER=hybrid_astar python main.py
   ```

4. Switch to **Parallel Parking**.
5. Switch the scenario from **Clear** to **Obstacle**.
6. Run `python evaluate.py` and use the CSV metrics in the final report.

## 8. Current Limitations

- Hybrid A* is functional but not optimized; parallel cases can take several seconds.
- The path is not smoothed yet.
- Obstacles are simple static rectangles, not sensor-derived dynamic obstacles.
- There is no CARLA integration yet.
- The baseline planner depends on SciPy.

## 9. Suggested Next Improvements

- Add harder obstacle layouts.
- Add path smoothing after Hybrid A*.
- Add a controller-tracking error metric.
- Add repeated trials over lane widths and spot sizes.
- Export evaluation results to CSV files.
- Integrate a CARLA parking-lot scene.

