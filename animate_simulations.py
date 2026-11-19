#!/usr/bin/env python3
"""Render MP4/GIF animations showing where counting errors arise."""

from __future__ import annotations

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from simulate_fuzzy_count import run_no_error_simulation

SCENARIOS = {
    "clean_baseline": {
        "white_noise_px": 0.0, "miss_probability": 0.0,
        "burst_miss_probability": 0.0, "max_burst_frames": 1,
        "false_positive_rate": 0.0,
    },
    "mild_location_jitter": {
        "white_noise_px": 0.8, "miss_probability": 0.0,
        "burst_miss_probability": 0.0, "max_burst_frames": 1,
        "false_positive_rate": 0.0,
    },
    "random_and_burst_drops": {
        "white_noise_px": 0.4, "miss_probability": 0.15,
        "burst_miss_probability": 0.02, "max_burst_frames": 12,
        "false_positive_rate": 0.0,
    },
    "fuzzy_robust_false_positives": {
        "white_noise_px": 0.8, "miss_probability": 0.05,
        "burst_miss_probability": 0.0, "max_burst_frames": 1,
        "false_positive_rate": 0.12,
    },
}


def render(name: str, settings: dict, output_dir: Path, stride: int = 6) -> tuple[Path, Path]:
    simulation_height = 100.0
    animation_settings = dict(settings)
    animation_settings["fish_spacing_px"] = 8.0

    summary, rows = run_no_error_simulation(
        fish_count=12,
        duration_s=18.0,
        fps=30.0,
        height=100.0,
        line_position=0.5,
        speed_px_s=7.0,
        arrival_interval_s=0.35,
        ma_window=30,
        random_seed=7,
        record_geometry=True,
        **animation_settings,
    )
    frames = rows[::stride]
    x_max = summary["simulation_width"]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8), facecolor="#0e131b")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], hspace=0.28, wspace=0.25)
    view = fig.add_subplot(grid[:, 0])
    count_ax = fig.add_subplot(grid[0, 1])
    info_ax = fig.add_subplot(grid[1, 1])

    view.set_xlim(0, x_max)
    view.set_ylim(105, -5)
    view.set_aspect("equal", adjustable="box")
    view.axhline(50, color="#ffd166", linewidth=2, linestyle="--")
    view.text(3, 47.5, "COUNTING LINE", color="#ffd166", fontsize=10, fontweight="bold")
    truth_scatter = view.scatter([], [], s=55, c="#55d6be", edgecolors="white", linewidths=0.4, label="Ground truth")
    detection_scatter = view.scatter([], [], s=100, facecolors="none", edgecolors="#ff6b6b", linewidths=1.4, label="Noisy detections")
    tracked_scatter = view.scatter([], [], s=38, marker="x", c="#6ea8fe", linewidths=2.0, label="Active ByteTrack")
    view.legend(loc="upper right", frameon=False)
    view.set_xlabel("Horizontal position (px)")
    view.set_ylabel("Vertical position (px)")
    view.set_title("Fish, detections, and tracker positions")
    view.grid(alpha=0.10)

    truth_line, = count_ax.plot([], [], color="#55d6be", linewidth=2.5, label="Truth")
    tracked_line, = count_ax.plot([], [], color="#ff6b6b", linewidth=2.2, label="Tracking line count")
    fuzzy_line, = count_ax.plot([], [], color="#ffd166", linewidth=2.2, label="Fuzzy flux")
    count_ax.set_xlim(0, rows[-1]["time_s"])
    final_truth = rows[-1]["actual_up"] + rows[-1]["actual_down"]
    max_count = max(final_truth, summary["tracked_up"] + summary["tracked_down"], summary["fuzzy_total"], 1.0) * 1.2
    count_ax.set_ylim(0, max_count)
    count_ax.set_xlabel("Time (s)")
    count_ax.set_ylabel("Cumulative crossings")
    count_ax.legend(frameon=False, fontsize=8)
    count_ax.grid(alpha=0.15)
    count_ax.set_title("Cumulative counting divergence")

    status = info_ax.text(0.02, 0.98, "", transform=info_ax.transAxes, va="top", fontsize=12,
                          family="monospace", linespacing=1.5)
    info_ax.axis("off")
    info_ax.set_title("Current error sources", fontweight="bold")
    title = fig.text(0.5, 0.97, "", ha="center", fontsize=17, fontweight="bold")

    all_times = np.asarray([row["time_s"] for row in rows])
    truth_counts = np.asarray([row["actual_up"] + row["actual_down"] for row in rows])
    tracked_counts = np.asarray([row["tracked_up"] + row["tracked_down"] for row in rows])
    fuzzy_counts = np.asarray([row["fuzzy_total"] for row in rows])

    def update(frame_number: int):
        row = frames[frame_number]
        truth = np.asarray([[item[1], item[2]] for item in row["truth_xy"]], dtype=float)
        detected = np.asarray(row["detection_xy"], dtype=float)
        tracked = np.asarray([[item[1], item[2]] for item in row["tracked_xy"]], dtype=float)
        truth_scatter.set_offsets(truth if len(truth) else np.empty((0, 2)))
        detection_scatter.set_offsets(detected if len(detected) else np.empty((0, 2)))
        tracked_scatter.set_offsets(tracked if len(tracked) else np.empty((0, 2)))

        current = row["frame"]
        truth_line.set_data(all_times[:current + 1], truth_counts[:current + 1])
        tracked_line.set_data(all_times[:current + 1], tracked_counts[:current + 1])
        fuzzy_line.set_data(all_times[:current + 1], fuzzy_counts[:current + 1])

        truth_total = row["actual_up"] + row["actual_down"]
        tracked_total = row["tracked_up"] + row["tracked_down"]
        status.set_text(
            f"TIME              {row['time_s']:5.1f} s\n\n"
            f"TRUTH CROSSINGS   {truth_total:6d}\n"
            f"TRACKED CROSSINGS {tracked_total:6d}  err {tracked_total - truth_total:+5d}\n"
            f"FUZZY CROSSINGS   {row['fuzzy_total']:6.1f}  err {row['fuzzy_total'] - truth_total:+5.1f}\n\n"
            f"TRUTH FISH        {row['truth_active_count']:6d}\n"
            f"DETECTIONS        {row['detected_count']:6d}\n"
            f"MISSED NOW        {row['missed_detection_count']:6d}\n"
            f"FALSE POSITIVES   {row['false_positive_count']:6d}\n"
            f"ACTIVE TRACKS     {row['tracked_active_count']:6d}\n"
            f"RECOVERABLE       {row['recoverable_track_count']:6d}"
        )
        title.set_text(name.replace("_", " ").title())
        return truth_scatter, detection_scatter, tracked_scatter, truth_line, tracked_line, fuzzy_line, status, title

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.gif"
    animator = animation.FuncAnimation(fig, update, frames=len(frames), interval=120, blit=True)
    mp4_path = output_dir / f"{name}.mp4"
    animator.save(mp4_path, writer=animation.FFMpegWriter(fps=8, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]), dpi=90)
    plt.close(fig)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4_path),
        "-vf", "fps=8,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(output_path),
    ], check=True)
    return mp4_path, output_path


def main() -> None:
    output_dir = Path("debug/counting_error_animations")
    for name, settings in SCENARIOS.items():
        paths = render(name, settings, output_dir)
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
