#!/usr/bin/env python3
"""Create per-sweep MAE ± SD error-bar plots and a numeric summary table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AXES = (
    "speed_px_s",
    "speed_variability_percent",
    "opposite_percent",
    "stationary_percent",
    "white_noise_px",
    "systematic_bias_px",
    "miss_probability",
    "burst_miss_probability",
    "max_burst_frames",
    "fish_spacing_px",
)
LABELS = {
    "speed_px_s": "Speed (px/s)",
    "speed_variability_percent": "Speed variability (%)",
    "opposite_percent": "Opposite-direction fish (%)",
    "stationary_percent": "Stationary fish (%)",
    "white_noise_px": "Detection-location σ (px)",
    "systematic_bias_px": "Systematic vertical bias (px)",
    "miss_probability": "Independent detection-drop rate",
    "burst_miss_probability": "Burst start rate",
    "max_burst_frames": "Maximum burst length (frames)",
    "fish_spacing_px": "Fish spacing (px)",
}
METHODS = {
    "Tracking by detection": "error_tracked_total_vs_all_crossings",
    "Fuzzy flux": "error_fuzzy_total_vs_all_crossings",
}
COLORS = {"Tracking by detection": "#ff6b6b", "Fuzzy flux": "#55d6be"}


def varying_axes(rows: list[dict]) -> list[str]:
    return [axis for axis in AXES if axis in rows[0] and len({row[axis] for row in rows}) > 1]


def grouped_stats(rows: list[dict], axis: str, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    buckets: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        actual_total = abs(float(row["actual_up"])) + abs(float(row["actual_down"]))
        denominator = max(actual_total, 1.0)
        error_rate = 100.0 * abs(float(row[field])) / denominator
        x_value = 100.0 * float(row[axis]) if axis in {"miss_probability", "burst_miss_probability"} else float(row[axis])
        buckets[x_value].append(error_rate)
    x = np.asarray(sorted(buckets), dtype=float)
    mean = np.asarray([np.mean(buckets[value]) for value in x])
    std = np.asarray([np.std(buckets[value], ddof=1) if len(buckets[value]) > 1 else 0.0 for value in x])
    count = np.asarray([len(buckets[value]) for value in x])
    return x, mean, std, count


def plot_sweep(path: Path, output_path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    axes = varying_axes(rows)
    if not axes:
        return []

    available_methods = {
        label: field for label, field in METHODS.items() if field in rows[0]
    }
    columns = min(3, len(axes))
    row_count = (len(axes) + columns - 1) // columns
    plt.style.use("dark_background")
    fig, axis_grid = plt.subplots(row_count, columns, figsize=(5.4 * columns, 4.2 * row_count),
                                  facecolor="#10141c", squeeze=False)
    summary_rows: list[dict] = []

    for panel, axis in enumerate(axes):
        ax = axis_grid.flat[panel]
        for method, field in available_methods.items():
            x, mean, std, counts = grouped_stats(rows, axis, field)
            ax.errorbar(
                x, mean, yerr=std, marker="o", capsize=4, linewidth=2.2,
                color=COLORS[method], label=method,
            )
            for value, mae, spread, count in zip(x, mean, std, counts):
                summary_rows.append({
                    "sweep": path.stem,
                    "axis": axis,
                    "axis_value": value,
                    "method": method,
                    "accuracy_percent": mae,
                    "sd_percent_points": spread,
                    "n": int(count),
                })
        ax.set_title(LABELS.get(axis, axis), fontweight="bold")
        ax.set_xlabel("Independent detection error rate (%)" if axis == "miss_probability" else
                      "Burst start error rate (%)" if axis == "burst_miss_probability" else
                      LABELS.get(axis, axis))
        ax.set_ylabel("Counting error rate (%)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, fontsize=8)

    for unused in axis_grid.flat[len(axes):]:
        unused.axis("off")

    fig.suptitle(f"{path.stem}: Counting error rate ± SD across all other settings", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             "Error rate = 100 × absolute crossing error / actual crossings; bars are percentage-point SD.",
             ha="center", color="#aab2c0", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean absolute error with error bars for every sweep CSV")
    parser.add_argument("input", nargs="*", type=Path, default=list(Path("debug").glob("*sweep.csv")))
    parser.add_argument("--output_dir", type=Path, default=Path("debug/sweep_error_plots"))
    args = parser.parse_args()

    input_paths = args.input or list(Path("debug").glob("*sweep.csv"))
    all_summaries: list[dict] = []
    for path in input_paths:
        if not path.exists():
            continue
        output_path = args.output_dir / f"{path.stem}_error_bars.png"
        try:
            summary = plot_sweep(path, output_path)
        except Exception as error:
            print(f"Skipped {path}: {error}")
            continue
        if summary:
            all_summaries.extend(summary)
            print(f"{output_path}: {len(summary)} points")

    summary_path = args.output_dir / "sweep_error_summary.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        if all_summaries:
            writer = csv.DictWriter(handle, fieldnames=list(all_summaries[0]))
            writer.writeheader()
            writer.writerows(all_summaries)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
