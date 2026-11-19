#!/usr/bin/env python3
"""Deterministic simulation for comparing fuzzy flux counts with line crossings.

The baseline scenario has no missed detections, false detections, or track failures:
all fish enter at the bottom and move upward in a 60-second video.

It can also mix upward fish, downward fish, and stationary fish by percentage.
In all cases, movement and line crossings are modeled without detection noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from infer_video_detr import _FuzzyCrossingEstimate
from tracker import BYTETracker, get_motion_vector


def run_no_error_simulation(
    fish_count: int = 10,
    duration_s: float = 60.0,
    fps: float = 30.0,
    height: float = 100.0,
    line_position: float = 0.5,
    speed_px_s: float = 5.0,
    speed_variability_percent: float = 0.0,
    arrival_interval_s: float = 1.0,
    ma_window: int = 30,
    opposite_percent: float = 0.0,
    stationary_percent: float = 0.0,
    white_noise_px: float = 0.0,
    systematic_bias_px: float = 0.0,
    miss_probability: float = 0.0,
    burst_miss_probability: float = 0.0,
    max_burst_frames: int = 1,
    false_positive_rate: float = 0.0,
    fish_spacing_px: float = 40.0,
    simulation_width: float | None = None,
    random_seed: int = 0,
    track_thresh: float = 0.25,
    record_geometry: bool = False,
) -> tuple[dict, list[dict]]:
    """Run a perfect-tracking simulation with mixed fish directions.

    Fish are assigned deterministically: IDs first are downward, the next are
    stationary, and the remainder are upward. Percentages are converted to fish
    counts and rounded to the nearest whole fish.
    """
    if fish_count < 1:
        raise ValueError("fish_count must be positive")
    if duration_s <= 0 or fps <= 0 or height <= 0:
        raise ValueError("duration, FPS, and height must be positive")
    if speed_px_s < 0:
        raise ValueError("speed_px_s must be non-negative")
    if ma_window < 1:
        raise ValueError("ma_window must be positive")
    if not 0.0 < line_position < 1.0:
        raise ValueError("line_position must be between 0 and 1")
    if not 0.0 <= opposite_percent <= 100.0:
        raise ValueError("opposite_percent must be between 0 and 100")
    if not 0.0 <= stationary_percent <= 100.0:
        raise ValueError("stationary_percent must be between 0 and 100")
    if opposite_percent + stationary_percent > 100.0:
        raise ValueError("opposite_percent and stationary_percent cannot sum above 100")
    if white_noise_px < 0:
        raise ValueError("white_noise_px must be non-negative")
    if not 0.0 <= miss_probability <= 1.0:
        raise ValueError("miss_probability must be between 0 and 1")
    if not 0.0 <= burst_miss_probability <= 1.0:
        raise ValueError("burst_miss_probability must be between 0 and 1")
    if max_burst_frames < 1:
        raise ValueError("max_burst_frames must be at least 1")
    if not 0.0 <= false_positive_rate <= 1.0:
        raise ValueError("false_positive_rate must be between 0 and 1")
    if fish_spacing_px < 1:
        raise ValueError("fish_spacing_px must be positive")

    width = height * 16.0 / 9.0 if simulation_width is None else simulation_width
    if width <= 0:
        raise ValueError("simulation_width must be positive")

    miss_remaining: dict[int, int] = {}

    rng = np.random.default_rng(random_seed)
    tracker = BYTETracker(track_thresh=track_thresh, track_buffer=ma_window, min_box_score=0.1)

    y_line = line_position * height
    frame_count = int(round(duration_s * fps))
    first_enter_s = 1.0 / fps

    opposite_count = int(fish_count * opposite_percent / 100.0 + 0.5)
    stationary_count = int(fish_count * stationary_percent / 100.0 + 0.5)
    upward_count = fish_count - opposite_count - stationary_count

    def direction(track_id: int) -> str:
        if track_id <= opposite_count:
            return "down"
        if track_id <= opposite_count + stationary_count:
            return "stationary"
        return "up"

    # Ground truth is tracked separately from the noisy detections. ByteTrack IDs
    # may split, merge, or be lost, so the tracked line count is an experimental
    # outcome rather than an alias for the truth count.
    fuzzy = _FuzzyCrossingEstimate(height, fps, ma_window)
    truth_last_side: dict[int, int] = {}
    track_last_side: dict[int, int] = {}
    count_up = 0
    count_down = 0
    tracked_count_up = 0
    tracked_count_down = 0
    rows: list[dict] = []

    fish_speeds: dict[int, float] = {}
    for track_id in range(1, fish_count + 1):
        if speed_variability_percent == 0:
            fish_speeds[track_id] = speed_px_s
        else:
            factor = 1.0 + ((track_id - (fish_count + 1) / 2.0) / max(fish_count - 1, 1)) * (speed_variability_percent / 50.0)
            fish_speeds[track_id] = max(0.0, speed_px_s * factor)

    for frame_index in range(frame_count):
        time_s = frame_index / fps
        active: list[tuple[int, float, str]] = []

        for track_id in range(1, fish_count + 1):
            fish_direction = direction(track_id)
            enter_s = first_enter_s + (track_id - 1) * arrival_interval_s
            elapsed_s = time_s - enter_s
            if elapsed_s < 0:
                continue

            speed = fish_speeds[track_id]
            if fish_direction == "up":
                y = height - speed * elapsed_s
            elif fish_direction == "down":
                y = 0.0 + speed * elapsed_s
            else:
                stationary_index = track_id - opposite_count - 1
                y = height * (0.25 + 0.5 * (stationary_index % 5) / 4.0)

            # Coordinated lateral schooling is added to x at detection time;
            # vertical velocity remains constant for an exact clean flux baseline.
            phase = (track_id % 6) * 0.7
            y += np.sin(elapsed_s * 0.45 + phase) * 0.75

            if y < 0 or y > height:
                continue

            active.append((track_id, y, fish_direction))

        motion_vectors: list[np.ndarray] = []

        for track_id, y, fish_direction in active:
            truth_side = -1 if y < y_line else (1 if y > y_line else 0)
            if truth_side != 0:
                previous_side = truth_last_side.get(track_id)
                if previous_side is not None:
                    if previous_side == 1 and truth_side == -1:
                        count_up += 1
                    elif previous_side == -1 and truth_side == 1:
                        count_down += 1
                truth_last_side[track_id] = truth_side

        detections: list[list[float]] = []
        box_w, box_h = 12.0, 12.0
        start_x = max(10.0, (width - (fish_count - 1) * fish_spacing_px) / 2.0)
        for track_id, y, _fish_direction in active:
            x = start_x + (track_id - 1) * fish_spacing_px
            group_x = 18.0 * np.sin(time_s * 0.22)
            local_x = 4.0 * np.sin(time_s * 0.8 + track_id * 0.9)
            x += group_x + local_x
            remaining = miss_remaining.get(track_id, 0)
            if remaining > 0:
                miss_remaining[track_id] = remaining - 1
                continue
            if rng.random() < burst_miss_probability:
                miss_remaining[track_id] = int(rng.integers(0, max_burst_frames + 1))
                continue
            if rng.random() < miss_probability:
                continue

            noisy_x = x + rng.normal(0.0, white_noise_px)
            noisy_y = y + systematic_bias_px + rng.normal(0.0, white_noise_px)
            detections.append([
                noisy_x - box_w / 2.0,
                noisy_y - box_h / 2.0,
                noisy_x + box_w / 2.0,
                noisy_y + box_h / 2.0,
                0.9,
            ])

        false_positive_count = 0
        if rng.random() < false_positive_rate:
            false_positive_count += 1
            false_x = float(rng.uniform(max(0.0, start_x - 20.0), min(width, start_x + fish_count * fish_spacing_px + 20.0)))
            false_y = float(y_line + rng.normal(0.0, max(1.0, white_noise_px + 1.0)))
            detections.append([
                false_x - box_w / 2.0,
                false_y - box_h / 2.0,
                false_x + box_w / 2.0,
                false_y + box_h / 2.0,
                0.9,
            ])

        active_tracks = tracker.update(detections)
        recoverable_tracks = active_tracks + tracker.lost_stracks
        active_track_ids = {track.track_id for track in active_tracks}
        for track in active_tracks:
            cy = float((track.tlbr[1] + track.tlbr[3]) / 2.0)
            side = -1 if cy < y_line else (1 if cy > y_line else 0)
            if side != 0:
                previous_side = track_last_side.get(track.track_id)
                if previous_side is not None:
                    if previous_side == 1 and side == -1:
                        tracked_count_up += 1
                    elif previous_side == -1 and side == 1:
                        tracked_count_down += 1
                track_last_side[track.track_id] = side

            motion = get_motion_vector(track, frame_index, fps)
            if motion is not None:
                motion_vectors.append(motion[0])

        for lost_track in tracker.lost_stracks:
            if lost_track.track_id in active_track_ids:
                continue
            motion = get_motion_vector(lost_track, frame_index, fps)
            if motion is not None:
                motion_vectors.append(motion[0])

        fuzzy_result = fuzzy.update(frame_index, len(recoverable_tracks), motion_vectors)
        if record_geometry:
            rows_geometry = {
                "truth_xy": [[track_id, float((track_id - 1) * fish_spacing_px + start_x +
                                               18.0 * np.sin(time_s * 0.22) +
                                               4.0 * np.sin(time_s * 0.8 + track_id * 0.9)), float(y)]
                             for track_id, y, _ in active],
                "detection_xy": [[float((d[0] + d[2]) / 2.0), float((d[1] + d[3]) / 2.0)] for d in detections],
                "tracked_xy": [[track.track_id, float((track.tlbr[0] + track.tlbr[2]) / 2.0),
                                 float((track.tlbr[1] + track.tlbr[3]) / 2.0)]
                                for track in active_tracks],
            }
        else:
            rows_geometry = None

        rows.append({
            "frame": frame_index,
            "time_s": time_s,
            "truth_active_count": len(active),
            "detected_count": len(detections),
            "missed_detection_count": max(0, len(active) - (len(detections) - false_positive_count)),
            "false_positive_count": false_positive_count,
            "tracked_active_count": len(active_tracks),
            "recoverable_track_count": len(recoverable_tracks),
            "velocity_sample_count": len(motion_vectors),
            **(rows_geometry or {}),
            "actual_up": count_up,
            "actual_down": count_down,
            "tracked_up": tracked_count_up,
            "tracked_down": tracked_count_down,
            "error_tracked_up_vs_actual_up": tracked_count_up - count_up,
            "error_tracked_down_vs_actual_down": tracked_count_down - count_down,
            "error_tracked_total_vs_all_crossings": (tracked_count_up + tracked_count_down) - (count_up + count_down),
            "fuzzy_up": fuzzy_result["up"],
            "fuzzy_down": fuzzy_result["down"],
            "fuzzy_total": fuzzy_result["total"],
            "fuzzy_net_down_minus_up": fuzzy_result["net"],
            "moving_average_count": fuzzy_result["moving_average_count"],
        })

    final = rows[-1]
    summary = {
        "scenario": "mixed_directions_with_detection_noise",
        "video_seconds": duration_s,
        "fps": fps,
        "frame_count": frame_count,
        "height_px": height,
        "line_position_fraction": line_position,
        "line_y_px": y_line,
        "fish_count": fish_count,
        "opposite_percent": opposite_percent,
        "stationary_percent": stationary_percent,
        "opposite_fish": opposite_count,
        "stationary_fish": stationary_count,
        "upward_fish": upward_count,
        "speed_px_s": speed_px_s,
        "speed_variability_percent": speed_variability_percent,
        "arrival_interval_s": arrival_interval_s,
        "ma_window_frames": ma_window,
        "white_noise_px": white_noise_px,
        "systematic_bias_px": systematic_bias_px,
        "miss_probability": miss_probability,
        "burst_miss_probability": burst_miss_probability,
        "max_burst_frames": max_burst_frames,
        "false_positive_rate": false_positive_rate,
        "fish_spacing_px": fish_spacing_px,
        "simulation_width": width,
        "random_seed": random_seed,
        "actual_up": count_up,
        "actual_down": count_down,
        "tracked_up": tracked_count_up,
        "tracked_down": tracked_count_down,
        "error_tracked_up_vs_actual_up": tracked_count_up - count_up,
        "error_tracked_down_vs_actual_down": tracked_count_down - count_down,
        "error_tracked_total_vs_all_crossings": (tracked_count_up + tracked_count_down) - (count_up + count_down),
        "fuzzy_up": final["fuzzy_up"],
        "fuzzy_down": final["fuzzy_down"],
        "fuzzy_total": final["fuzzy_total"],
        "fuzzy_net_down_minus_up": final["fuzzy_net_down_minus_up"],
        "error_fuzzy_up_vs_actual_up": final["fuzzy_up"] - count_up,
        "error_fuzzy_down_vs_actual_down": final["fuzzy_down"] - count_down,
        "error_fuzzy_total_vs_all_crossings": final["fuzzy_total"] - (count_up + count_down),
    }
    return summary, rows


def run_percentage_sweep(
    percentages: list[int],
    speeds_px_s: list[float],
    speed_variabilities: list[float],
    white_noise_levels: list[float],
    systematic_bias_levels: list[float],
    miss_probability_levels: list[float],
    burst_probability_levels: list[float],
    burst_frame_levels: list[int],
    spacing_levels: list[float],
    random_seeds: list[int],
    fish_count: int,
    duration_s: float,
    fps: float,
    height: float,
    line_position: float,
    arrival_interval_s: float,
    ma_window: int,
    workers: int | None = None,
    false_positive_levels: list[float] | None = None,
) -> list[dict]:
    false_positive_levels = false_positive_levels or [0.0]
    scenarios = [
        {
            "fish_count": fish_count,
            "duration_s": duration_s,
            "fps": fps,
            "height": height,
            "line_position": line_position,
            "speed_px_s": speed_px_s,
            "speed_variability_percent": speed_variability,
            "arrival_interval_s": arrival_interval_s,
            "ma_window": ma_window,
            "opposite_percent": float(opposite_percent),
            "stationary_percent": float(stationary_percent),
            "white_noise_px": white_noise_px,
            "systematic_bias_px": systematic_bias_px,
            "miss_probability": miss_probability,
            "burst_miss_probability": burst_probability,
            "max_burst_frames": max_burst_frames,
            "false_positive_rate": false_positive_rate,
            "fish_spacing_px": fish_spacing_px,
            "random_seed": random_seed,
        }
        for speed_px_s in speeds_px_s
        for speed_variability in speed_variabilities
        for stationary_percent in percentages
        for opposite_percent in percentages
        for white_noise_px in white_noise_levels
        for systematic_bias_px in systematic_bias_levels
        for miss_probability in miss_probability_levels
        for burst_probability in burst_probability_levels
        for max_burst_frames in burst_frame_levels
        for false_positive_rate in false_positive_levels
        for fish_spacing_px in spacing_levels
        for random_seed in random_seeds
        if opposite_percent + stationary_percent <= 100
    ]

    def run_scenario(scenario: dict) -> dict:
        summary, _ = run_no_error_simulation(**scenario)
        return summary

    max_workers = workers or min(32, (os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(run_scenario, scenarios))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for --plot")

    times = [row["time_s"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(times, [row["truth_active_count"] for row in rows], label="Ground-truth fish")
    axes[0].plot(times, [row["tracked_active_count"] for row in rows], label="Tracked fish")
    axes[0].set_ylabel("Fish")
    axes[0].legend()
    axes[0].set_title("Perfect-tracking baseline: bottom-to-top fish")

    axes[1].plot(times, [row["actual_up"] for row in rows], label="Actual line: up")
    axes[1].plot(times, [row["fuzzy_up"] for row in rows], label="Fuzzy: up", linestyle="--")
    axes[1].plot(times, [row["actual_down"] for row in rows], label="Actual line: down")
    axes[1].plot(times, [row["fuzzy_down"] for row in rows], label="Fuzzy: down", linestyle="--")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Cumulative crossings")
    axes[1].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_sweep_plot(path: Path, summaries: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise RuntimeError("matplotlib is required for --sweep_plot_output")

    def mean_for(filters: dict, field: str) -> float:
        values = [abs(float(item[field])) if field.startswith("abs_") else float(item[field])
                  for item in summaries
                  if all(float(item[key]) == float(value) for key, value in filters.items())]
        return float(np.mean(values)) if values else 0.0

    for item in summaries:
        item["abs_error_tracked_total_vs_all_crossings"] = abs(float(item["error_tracked_total_vs_all_crossings"]))
        item["abs_error_fuzzy_total_vs_all_crossings"] = abs(float(item["error_fuzzy_total_vs_all_crossings"]))

    def grouped_mean(axis: str, field: str) -> tuple[list[float], list[float]]:
        values = sorted({float(item[axis]) for item in summaries})
        means = [float(np.mean([
            float(row[field]) for row in summaries if float(row[axis]) == value
        ])) for value in values]
        return values, means

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#10141c")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.08], hspace=0.34, wspace=0.28)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])]
    heat_tracks = fig.add_subplot(grid[1, :2])
    heat_fuzzy = fig.add_subplot(grid[1, 2])

    colors = {"Tracking by detection": "#ff6b6b", "Fuzzy flux": "#55d6be"}

    def effect_plot(ax, axis: str, label: str):
        x, tracked = grouped_mean(axis, "abs_error_tracked_total_vs_all_crossings")
        _, fuzzy = grouped_mean(axis, "abs_error_fuzzy_total_vs_all_crossings")
        ax.plot(x, tracked, "o-", color=colors["Tracking by detection"], linewidth=2.5,
                label="Tracking by detection")
        ax.plot(x, fuzzy, "o-", color=colors["Fuzzy flux"], linewidth=2.5, label="Fuzzy flux")
        ax.fill_between(x, 0, tracked, color=colors["Tracking by detection"], alpha=0.08)
        ax.fill_between(x, 0, fuzzy, color=colors["Fuzzy flux"], alpha=0.08)
        ax.set_xlabel(label)
        ax.set_ylabel("Mean absolute total error")
        ax.set_ylim(bottom=0.0)
        ax.margins(x=0.04, y=0.08)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)

    effect_plot(axes[0], "white_noise_px", "Detection location σ (px)")
    effect_plot(axes[1], "fish_spacing_px", "Horizontal fish spacing (px)")
    effect_plot(axes[2], "miss_probability", "Independent detection-drop rate")
    axes[0].set_title("Localization noise dominates", fontweight="bold")
    axes[1].set_title("Crowding / identity association", fontweight="bold")
    axes[2].set_title("Random detection loss", fontweight="bold")

    noise = sorted({float(item["white_noise_px"]) for item in summaries})
    spacing = sorted({float(item["fish_spacing_px"]) for item in summaries})
    tracked_matrix = np.array([[
        mean_for({"white_noise_px": n, "fish_spacing_px": s}, "abs_error_tracked_total_vs_all_crossings")
        for s in spacing
    ] for n in noise])
    fuzzy_matrix = np.array([[
        mean_for({"white_noise_px": n, "fish_spacing_px": s}, "abs_error_fuzzy_total_vs_all_crossings")
        for s in spacing
    ] for n in noise])
    shared_max = max(tracked_matrix.max(), fuzzy_matrix.max())

    im = heat_tracks.imshow(tracked_matrix, cmap="magma", vmin=0, vmax=shared_max, aspect="auto", origin="lower")
    heat_tracks.set_title("Tracking error — location noise × crowding", fontweight="bold")
    heat_tracks.set_xticks(range(len(spacing)), [f"{s:g}" for s in spacing])
    heat_tracks.set_yticks(range(len(noise)), [f"{n:g}" for n in noise])
    heat_tracks.set_xlabel("Fish spacing (px)")
    heat_tracks.set_ylabel("Detection location σ (px)")
    for i in range(len(noise)):
        for j in range(len(spacing)):
            heat_tracks.text(j, i, f"{tracked_matrix[i, j]:.0f}", ha="center", va="center",
                             color="white" if tracked_matrix[i, j] > shared_max * 0.35 else "#111")
    fig.colorbar(im, ax=heat_tracks, fraction=0.025, pad=0.02, label="Mean absolute crossings")

    im2 = heat_fuzzy.imshow(fuzzy_matrix, cmap="magma", vmin=0, vmax=shared_max, aspect="auto", origin="lower")
    heat_fuzzy.set_title("Fuzzy error — same scale", fontweight="bold")
    heat_fuzzy.set_xticks(range(len(spacing)), [f"{s:g}" for s in spacing])
    heat_fuzzy.set_yticks(range(len(noise)), [f"{n:g}" for n in noise])
    heat_fuzzy.set_xlabel("Fish spacing (px)")
    heat_fuzzy.set_ylabel("Detection location σ (px)")
    for i in range(len(noise)):
        for j in range(len(spacing)):
            heat_fuzzy.text(j, i, f"{fuzzy_matrix[i, j]:.0f}", ha="center", va="center",
                            color="white" if fuzzy_matrix[i, j] > shared_max * 0.35 else "#111")
    fig.colorbar(im2, ax=heat_fuzzy, fraction=0.045, pad=0.03, label="Mean absolute crossings")

    tracked_bias = float(np.mean([float(item["error_tracked_total_vs_all_crossings"]) for item in summaries]))
    fuzzy_bias = float(np.mean([float(item["error_fuzzy_total_vs_all_crossings"]) for item in summaries]))
    fig.suptitle(
        "Detection-noise stress test: fuzzy flux vs. decision-line tracking\n"
        f"Mean signed bias: tracking {tracked_bias:+.1f} crossings  ·  fuzzy {fuzzy_bias:+.1f} crossings",
        fontsize=17, fontweight="bold", y=0.99,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: dict) -> None:
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fuzzy flux counts with decision-line crossings")
    parser.add_argument("--fish_count", type=int, default=10)
    parser.add_argument("--duration_s", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--height", type=float, default=100.0)
    parser.add_argument("--line_position", type=float, default=0.5,
                        help="Horizontal line position as a fraction of frame height")
    parser.add_argument("--speed_px_s", type=float, default=5.0)
    parser.add_argument("--speed_variability_percent", type=float, default=0.0,
                        help="Percentage spread of fish speeds around --speed_px_s")
    parser.add_argument("--arrival_interval_s", type=float, default=1.0)
    parser.add_argument("--ma_window", type=int, default=30)
    parser.add_argument("--opposite_percent", type=float, default=0.0,
                        help="Percentage of fish moving top-to-bottom")
    parser.add_argument("--stationary_percent", type=float, default=0.0,
                        help="Percentage of fish remaining still")
    parser.add_argument("--sweep_percent", action="append", type=int,
                        help="Percentage value for a sweep grid; repeat to add values")
    parser.add_argument("--sweep_speed", action="append", type=float,
                        help="Speed value for a sweep grid; repeat to add values")
    parser.add_argument("--sweep_speed_variability", action="append", type=float,
                        help="Speed variability percentage for a sweep; repeat to add values")
    parser.add_argument("--workers", type=int, default=None,
                        help="Maximum worker threads for parallel scenario simulation")
    parser.add_argument("--csv_output", type=Path,
                        default=Path("debug/fuzzy_count_no_errors.csv"))
    parser.add_argument("--json_output", type=Path, default=None)
    parser.add_argument("--plot_output", type=Path, default=None)
    parser.add_argument("--white_noise_px", type=float, default=0.0,
                        help="Zero-mean Gaussian detection jitter standard deviation in pixels")
    parser.add_argument("--systematic_bias_px", type=float, default=0.0,
                        help="Fixed vertical bias added to every detection")
    parser.add_argument("--sweep_white_noise", action="append", type=float,
                        help="White-noise standard deviation to sweep; repeat to add values")
    parser.add_argument("--sweep_systematic_bias", action="append", type=float,
                        help="Systematic vertical bias to sweep; repeat to add values")
    parser.add_argument("--sweep_seed", action="append", type=int,
                        help="Random seed to sweep; repeat to add values")
    parser.add_argument("--miss_probability", type=float, default=0.0,
                        help="Independent per-fish, per-frame probability of a missing detection")
    parser.add_argument("--burst_miss_probability", type=float, default=0.0,
                        help="Probability of starting a correlated multi-frame dropout")
    parser.add_argument("--max_burst_frames", type=int, default=1,
                        help="Maximum additional missed frames after a burst starts")
    parser.add_argument("--sweep_miss_probability", action="append", type=float,
                        help="Independent miss probability to sweep")
    parser.add_argument("--sweep_burst_probability", action="append", type=float,
                        help="Dropout-burst start probability to sweep")
    parser.add_argument("--sweep_burst_frames", action="append", type=int,
                        help="Maximum burst length to sweep")
    parser.add_argument("--fish_spacing_px", type=float, default=40.0,
                        help="Horizontal distance between fish centroids in pixels")
    parser.add_argument("--sweep_spacing", action="append", type=float,
                        help="Horizontal fish spacing to sweep; repeat to add values")
    parser.add_argument("--sweep_plot_output", type=Path, default=None,
                        help="Write an aggregate error heatmap and sensitivity plot for a sweep")
    args = parser.parse_args()

    if args.sweep_percent or args.sweep_speed or args.sweep_white_noise or args.sweep_systematic_bias or args.sweep_miss_probability or args.sweep_burst_probability or args.sweep_burst_frames or args.sweep_spacing:
        percentages = sorted(set(args.sweep_percent)) if args.sweep_percent else [0]
        speeds_px_s = sorted(set(args.sweep_speed)) if args.sweep_speed else [args.speed_px_s]
        speed_variabilities = sorted(set(args.sweep_speed_variability)) if args.sweep_speed_variability else [args.speed_variability_percent]
        white_noise_levels = sorted(set(args.sweep_white_noise)) if args.sweep_white_noise else [args.white_noise_px]
        systematic_bias_levels = sorted(set(args.sweep_systematic_bias)) if args.sweep_systematic_bias else [args.systematic_bias_px]
        miss_probability_levels = sorted(set(args.sweep_miss_probability)) if args.sweep_miss_probability else [args.miss_probability]
        burst_probability_levels = sorted(set(args.sweep_burst_probability)) if args.sweep_burst_probability else [args.burst_miss_probability]
        burst_frame_levels = sorted(set(args.sweep_burst_frames)) if args.sweep_burst_frames else [args.max_burst_frames]
        spacing_levels = sorted(set(args.sweep_spacing)) if args.sweep_spacing else [args.fish_spacing_px]
        random_seeds = sorted(set(args.sweep_seed)) if args.sweep_seed else [0]
        results = run_percentage_sweep(
            percentages=percentages,
            speeds_px_s=speeds_px_s,
            speed_variabilities=speed_variabilities,
            white_noise_levels=white_noise_levels,
            systematic_bias_levels=systematic_bias_levels,
            miss_probability_levels=miss_probability_levels,
            burst_probability_levels=burst_probability_levels,
            burst_frame_levels=burst_frame_levels,
            spacing_levels=spacing_levels,
            random_seeds=random_seeds,
            fish_count=args.fish_count,
            duration_s=args.duration_s,
            fps=args.fps,
            height=args.height,
            line_position=args.line_position,
            arrival_interval_s=args.arrival_interval_s,
            ma_window=args.ma_window,
            workers=args.workers,
        )
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(results[0].keys())
        with args.csv_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(results, indent=2) + "\n")
        if args.sweep_plot_output is not None:
            write_sweep_plot(args.sweep_plot_output, results)
        print(json.dumps(results, indent=2))
        print(f"Sweep CSV: {args.csv_output}")
        return

    summary, rows = run_no_error_simulation(
        fish_count=args.fish_count,
        duration_s=args.duration_s,
        fps=args.fps,
        height=args.height,
        line_position=args.line_position,
        speed_px_s=args.speed_px_s,
        speed_variability_percent=args.speed_variability_percent,
        arrival_interval_s=args.arrival_interval_s,
        ma_window=args.ma_window,
        opposite_percent=args.opposite_percent,
        stationary_percent=args.stationary_percent,
        white_noise_px=args.white_noise_px,
        systematic_bias_px=args.systematic_bias_px,
        miss_probability=args.miss_probability,
        burst_miss_probability=args.burst_miss_probability,
        max_burst_frames=args.max_burst_frames,
        fish_spacing_px=args.fish_spacing_px,
    )

    write_csv(args.csv_output, rows)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot_output is not None:
        write_plot(args.plot_output, rows)

    print_summary(summary)
    print(f"CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
