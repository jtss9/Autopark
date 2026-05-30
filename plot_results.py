"""
Render report-ready figures from evaluator CSV files.

Usage:
    python plot_results.py results/main.csv --outdir results/figures

Produces:
    success_rate_by_planner.png     - bar chart of success rate per planner
    planning_time_by_planner.png    - boxplot/strip of planning time per planner
    path_length_by_planner.png      - bar chart of average path length
    sweep_<param>_<metric>.png      - line plot of metric vs swept parameter
    tracking_error.png              - cross-track error per planner (if tracked)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        return matplotlib
    except ImportError as exc:
        print(
            "plot_results: matplotlib not installed. "
            "Install with `pip install matplotlib`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes")


def _parse_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _by_planner(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        out[r.get("planner", "unknown")].append(r)
    return out


def plot_success_rate(rows, outpath: Path) -> None:
    import matplotlib.pyplot as plt
    groups = _by_planner(rows)
    planners = sorted(groups.keys())
    rates = []
    for p in planners:
        succ = sum(1 for r in groups[p] if _parse_bool(r.get("success", "")))
        rates.append(100.0 * succ / max(1, len(groups[p])))

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(planners, rates, color=["#2563eb", "#16a34a", "#f59e0b", "#9333ea"][: len(planners)])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Parking success rate by planner")
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_planning_time(rows, outpath: Path) -> None:
    import matplotlib.pyplot as plt
    groups = _by_planner(rows)
    planners = sorted(groups.keys())
    data = [
        [_parse_float(r.get("planning_time_s", "")) for r in groups[p]
         if _parse_float(r.get("planning_time_s", "")) > 0]
        for p in planners
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=planners, showmeans=True)
    ax.set_ylabel("Planning time (s)")
    ax.set_title("Planning time distribution by planner")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_path_length(rows, outpath: Path) -> None:
    import matplotlib.pyplot as plt
    groups = _by_planner(rows)
    planners = sorted(groups.keys())
    means: List[float] = []
    stds: List[float] = []
    for p in planners:
        vals = [
            _parse_float(r.get("path_length_m", ""))
            for r in groups[p]
            if _parse_bool(r.get("success", ""))
            and _parse_float(r.get("path_length_m", "")) > 0
        ]
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        else:
            mean, std = 0.0, 0.0
        means.append(mean)
        stds.append(std)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(planners, means, yerr=stds, color=["#2563eb", "#16a34a", "#f59e0b", "#9333ea"][: len(planners)], capsize=6)
    ax.set_ylabel("Mean path length (m, successful runs only)")
    ax.set_title("Path length by planner")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_sweep(
    rows: List[Dict[str, str]],
    sweep_key: str,
    metric: str,
    metric_label: str,
    outpath: Path,
) -> None:
    """Aggregate per (planner, x) value: mean line, individual points scattered."""
    import matplotlib.pyplot as plt
    series: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("sweep") != sweep_key:
            continue
        x = _parse_float(r.get(sweep_key, ""))
        y = _parse_float(r.get(metric, ""))
        if x != x or y != y:
            continue
        series[r.get("planner", "?")][x].append(y)
    if not series:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea"]
    for ci, (planner, xy) in enumerate(sorted(series.items())):
        color = colors[ci % len(colors)]
        xs_sorted = sorted(xy.keys())
        means = [sum(xy[x]) / len(xy[x]) for x in xs_sorted]
        ax.plot(xs_sorted, means, marker="o", color=color, label=f"{planner} (mean)")
        # Scatter individual samples lightly
        for x in xs_sorted:
            for y in xy[x]:
                ax.scatter([x], [y], color=color, alpha=0.25, s=18)
    ax.set_xlabel(sweep_key)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs {sweep_key}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_tracking_error(rows, outpath: Path) -> None:
    import matplotlib.pyplot as plt
    groups = _by_planner(rows)
    planners = sorted(groups.keys())
    have_tracker = False
    data_mean = []
    data_max = []
    for p in planners:
        means = [
            _parse_float(r.get("mean_cte_m", ""))
            for r in groups[p]
            if r.get("mean_cte_m", "").strip()
        ]
        maxs = [
            _parse_float(r.get("max_cte_m", ""))
            for r in groups[p]
            if r.get("max_cte_m", "").strip()
        ]
        means = [v for v in means if v == v]
        maxs = [v for v in maxs if v == v]
        if means:
            have_tracker = True
        data_mean.append(means)
        data_max.append(maxs)
    if not have_tracker:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(data_mean, tick_labels=planners, showmeans=True)
    axes[0].set_ylabel("Mean cross-track error (m)")
    axes[0].set_title("Pure-Pursuit tracking: mean CTE")
    axes[1].boxplot(data_max, tick_labels=planners, showmeans=True)
    axes[1].set_ylabel("Max cross-track error (m)")
    axes[1].set_title("Pure-Pursuit tracking: max CTE")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV file produced by evaluate.py")
    parser.add_argument(
        "--outdir",
        default="results/figures",
        help="Directory to write PNG figures (default: results/figures)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _try_import_matplotlib()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"plot_results: csv not found: {csv_path}", file=sys.stderr)
        return 1
    rows = _read_csv(csv_path)
    if not rows:
        print(f"plot_results: csv has no rows: {csv_path}", file=sys.stderr)
        return 1

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    plot_success_rate(rows, out / f"{stem}_success_rate.png")
    plot_planning_time(rows, out / f"{stem}_planning_time.png")
    plot_path_length(rows, out / f"{stem}_path_length.png")
    plot_tracking_error(rows, out / f"{stem}_tracking_error.png")

    # Generate sweep figures for any sweep column that's present.
    sweeps = sorted({r.get("sweep") for r in rows if r.get("sweep") not in (None, "", "none")})
    for sweep_key in sweeps:
        for metric, label in (
            ("planning_time_s", "Planning time (s)"),
            ("path_length_m", "Path length (m)"),
            ("final_pos_error_m", "Final position error (m)"),
        ):
            plot_sweep(rows, sweep_key, metric, label,
                       out / f"{stem}_sweep_{sweep_key}_{metric}.png")

    print(f"plot_results: wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
