"""Render RL trajectory JSONs (results/rl_figure_paths.json) into top-down
figure images: lane, spot, obstacle, trajectory, car footprints.

Usage:  python src/plot_rl_paths.py [path/to/paths.json]
Writes results/<key>.png for every entry in the JSON.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


def car_corners(x, y, theta, length, width):
    c, s = math.cos(theta), math.sin(theta)
    hw = width / 2
    pts = [(0, hw), (length, hw), (length, -hw), (0, -hw)]
    return [(x + dx * c - dy * s, y + dx * s + dy * c) for dx, dy in pts]


def render(key, d, out_dir):
    env = d["env"]
    traj = d["trajectory_xytheta"]
    lane = env["lane_rect_xywh"]
    spot = env["spot_rect_xywh"]
    L, W = env["car_length"], env["car_width"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.add_patch(Rectangle((lane[0], lane[1]), lane[2], lane[3],
                           fc="#e8e8e8", ec="#888", lw=1, zorder=0))
    ax.add_patch(Rectangle((spot[0], spot[1]), spot[2], spot[3],
                           fc="#dff0d8", ec="#3a7d3a", lw=1.6, zorder=1))
    if env.get("obstacle_xywh"):
        ob = env["obstacle_xywh"]
        ax.add_patch(Rectangle((ob[0], ob[1]), ob[2], ob[3],
                               fc="#c0392b", ec="#7b241c", lw=1.2, zorder=3))

    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    ax.plot(xs, ys, "-", color="#1f5fbf", lw=1.6, zorder=4,
            label="rear-axle path")

    # Car footprints: start (blue), a few intermediates (faint), final
    # (green if success, orange/red otherwise).
    n = len(traj)
    idxs = [0] + [int(n * f) for f in (0.25, 0.5, 0.75)] + [n - 1]
    final_color = {"success": "#2e8b57", "collision": "#c0392b"}.get(
        d["outcome"], "#e67e22")
    for k, i in enumerate(sorted(set(idxs))):
        x, y, th = traj[i]
        last = (i == n - 1)
        ax.add_patch(Polygon(car_corners(x, y, th, L, W), closed=True,
                             fc="none",
                             ec=final_color if last else "#1f5fbf",
                             lw=2.0 if last or i == 0 else 0.9,
                             alpha=1.0 if last or i == 0 else 0.45,
                             zorder=5))

    gx, gy, gth = env["goal_pose"]
    ax.plot([gx], [gy], "*", color="#3a7d3a", ms=13, zorder=6, label="goal")
    ax.arrow(gx, gy, 0.8 * math.cos(gth), 0.8 * math.sin(gth),
             head_width=0.18, color="#3a7d3a", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(lane[0] - 0.5, lane[0] + lane[2] + 0.5)
    lo = min(lane[1], spot[1]) - 0.5
    hi = max(lane[1] + lane[3], spot[1] + spot[3]) + 0.5
    ax.set_ylim(lo, hi)
    ax.set_title(f"{key}  ({d['outcome']}, {d['metrics']['steps']} steps)",
                 fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    out = os.path.join(out_dir, f"{key}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/rl_figure_paths.json"
    with open(src) as f:
        data = json.load(f)
    if "rollouts" in data:  # rl_obstacle_rollouts.json format
        data = {f"rollout_{i}": r for i, r in enumerate(data["rollouts"])}
    out_dir = os.path.dirname(src) or "."
    for key, d in data.items():
        if not isinstance(d, dict) or "trajectory_xytheta" not in d:
            continue
        print("wrote", render(key, d, out_dir))


if __name__ == "__main__":
    main()
