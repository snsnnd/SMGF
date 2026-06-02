from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def _load_case(case_dir: Path) -> dict:
    with (case_dir / "metadata.json").open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    trajectory = pd.read_csv(case_dir / "trajectory.csv")
    state = pd.read_csv(case_dir / "state_curves.csv")
    target = pd.read_csv(case_dir / "target_curves.csv")
    obstacles = pd.read_csv(case_dir / "obstacles.csv")
    with (case_dir / "metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    return {
        "metadata": metadata,
        "trajectory": trajectory,
        "state": state,
        "target": target,
        "obstacles": obstacles,
        "metrics": metrics,
    }


def _truncate_by_time(df: pd.DataFrame, end_time: float) -> pd.DataFrame:
    if "time" not in df.columns:
        return df
    return df[df["time"] <= end_time].copy()


def _plot_trajectory(ax, case: dict, title: str, show_final_ring: bool = False) -> None:
    metadata = case["metadata"]
    trajectory = _truncate_by_time(case["trajectory"], metadata["recommended_plot_end_time"])
    target = _truncate_by_time(case["target"], metadata["recommended_plot_end_time"])
    obstacles = case["obstacles"]

    for _, obs in obstacles.iterrows():
        circle = plt.Circle((obs["center_x"], obs["center_y"]), obs["radius"], color="gray", alpha=0.25)
        ax.add_patch(circle)

    for agent, group in trajectory.groupby("agent"):
        ax.plot(group["x"], group["y"], linewidth=1.4)
        ax.scatter(group.iloc[0]["x"], group.iloc[0]["y"], s=18, marker="o")
        ax.scatter(group.iloc[-1]["x"], group.iloc[-1]["y"], s=24, marker="x")

    if show_final_ring and len(trajectory["agent"].unique()) >= 3 and not target.empty:
        final_step = trajectory["step"].max()
        final = trajectory[trajectory["step"] == final_step].copy()
        target_final = target[target["step"] == target["step"].max()].iloc[0]
        center = np.array([target_final["target_x"], target_final["target_y"]], dtype=float)
        final["angle"] = np.mod(np.arctan2(final["y"] - center[1], final["x"] - center[0]), 2 * np.pi)
        final = final.sort_values("angle")
        ring_x = final["x"].tolist() + [final.iloc[0]["x"]]
        ring_y = final["y"].tolist() + [final.iloc[0]["y"]]
        ax.plot(ring_x, ring_y, color="tab:purple", linewidth=1.5, linestyle="-.", label="final ring")

    ax.plot(target["target_x"], target["target_y"], "k--", linewidth=2.0, label="target")
    ax.plot(target["pred_target_x"], target["pred_target_y"], color="tab:red", linestyle=":", linewidth=1.4, label="predicted")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def _plot_zoomed_trajectory(ax, case: dict, xlim: tuple[float, float], ylim: tuple[float, float], title: str) -> None:
    metadata = case["metadata"]
    trajectory = _truncate_by_time(case["trajectory"], metadata["recommended_plot_end_time"])
    target = _truncate_by_time(case["target"], metadata["recommended_plot_end_time"])
    obstacles = case["obstacles"]

    for _, obs in obstacles.iterrows():
        circle = plt.Circle((obs["center_x"], obs["center_y"]), obs["radius"], color="gray", alpha=0.25)
        ax.add_patch(circle)

    for _, group in trajectory.groupby("agent"):
        ax.plot(group["x"], group["y"], linewidth=1.2)
        ax.scatter(group.iloc[-1]["x"], group.iloc[-1]["y"], s=16, marker="x")

    ax.plot(target["target_x"], target["target_y"], "k--", linewidth=1.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)


def _plot_state_curves(axs, case: dict, title: str) -> None:
    metadata = case["metadata"]
    state = _truncate_by_time(case["state"], metadata["recommended_plot_end_time"])
    mean_state = state.groupby("time").agg(
        omega_mean=("omega", "mean"),
        omega_min=("omega", "min"),
        omega_max=("omega", "max"),
        rho_mean=("rho", "mean"),
        rho_min=("rho", "min"),
        rho_max=("rho", "max"),
        psi_mean=("psi_tilde", "mean"),
        psi_min=("psi_tilde", "min"),
        psi_max=("psi_tilde", "max"),
    ).reset_index()

    series = [
        ("psi", "psi_mean", "psi_min", "psi_max", "tab:blue"),
        ("omega", "omega_mean", "omega_min", "omega_max", "tab:orange"),
        ("rho", "rho_mean", "rho_min", "rho_max", "tab:green"),
    ]
    for ax, (label, mean_key, min_key, max_key, color) in zip(axs, series):
        ax.plot(mean_state["time"], mean_state[mean_key], color=color, linewidth=1.5)
        ax.fill_between(mean_state["time"], mean_state[min_key], mean_state[max_key], color=color, alpha=0.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)

    t_end = float(mean_state["time"].max())
    stage_marks = [
        (0.18 * t_end, "approach"),
        (0.52 * t_end, "high-pressure"),
        (0.82 * t_end, "recovery"),
    ]
    for ax in axs:
        for stage_time, _ in stage_marks:
            ax.axvline(stage_time, color="0.35", linestyle="--", linewidth=0.9, alpha=0.7)
    top_ax = axs[0]
    y_top = float(mean_state["psi_max"].max()) if not mean_state.empty else 1.0
    for stage_time, label in stage_marks:
        top_ax.text(stage_time, y_top + 0.04, label, ha="center", va="bottom", fontsize=8)
    axs[-1].set_xlabel("time [s]")
    axs[0].set_title(title)


def _plot_c_group_bar(outputs_root: Path, figure_dir: Path) -> None:
    summary = pd.read_csv(outputs_root / "c_group_candidate_full_compare_v2" / "summary_metrics.csv")
    summary = summary[summary["method"].isin(["M4", "M5", "M6", "M7"])]
    scenes = ["s4_narrow_passage_easy", "s4_narrow_passage_medium", "s4_narrow_passage_hard"]
    method_order = ["M4", "M5", "M6", "M7"]
    labels = ["easy", "medium", "hard"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.18
    xs = range(len(scenes))
    for idx, method in enumerate(method_order):
        sub = summary[summary["method"] == method].set_index("scene")
        axes[0].bar([x + (idx - 1.5) * width for x in xs], [sub.loc[s, "success_rate"] for s in scenes], width=width, label=method)
        axes[1].bar([x + (idx - 1.5) * width for x in xs], [sub.loc[s, "agent_collision_rate"] for s in scenes], width=width, label=method)

    axes[0].set_title("C-group success rate")
    axes[1].set_title("C-group agent collision rate")
    for ax in axes:
        ax.set_xticks(list(xs), labels)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "C_group_summary.png", dpi=200)
    plt.close(fig)


def _plot_d1_trend(outputs_root: Path, figure_dir: Path) -> None:
    summary = pd.read_csv(outputs_root / "phase1_validation_v2" / "D1_prediction_scan" / "summary_metrics.csv")
    summary = summary[summary["method"].isin(["M4", "M7", "M9"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("completion_time_mean", "completion time"),
        ("current_center_error_mean", "current center error"),
        ("gmax_mean", "gmax [deg]"),
        ("radius_error_final_mean", "radius error"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for method, group in summary.groupby("method"):
            ax.plot(group["t_pred"], group[metric], marker="o", linewidth=1.5, label=method)
        ax.set_title(title)
        ax.set_xlabel("T_p")
        ax.grid(True, alpha=0.2)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "D1_prediction_trends.png", dpi=200)
    plt.close(fig)


def plot_phase1_figures(repo_root: Path, data_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (data_root / "bundle_manifest.json").open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    case_map = {entry["figure_key"]: _load_case(repo_root / entry["path"]) for entry in manifest["entries"]}

    # A2 trajectory
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_trajectory(ax, case_map["A2_obstacle_avoidance"], "A2 obstacle avoidance")
    fig.tight_layout()
    fig.savefig(output_root / "A2_obstacle_avoidance.png", dpi=200)
    plt.close(fig)

    # B3 comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_trajectory(axes[0], case_map["B3_open_encirclement_m7"], "B3 open encirclement | M7", show_final_ring=True)
    _plot_trajectory(axes[1], case_map["B3_open_encirclement_m9"], "B3 open encirclement | M9", show_final_ring=True)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / "B3_open_encirclement_compare.png", dpi=200)
    plt.close(fig)

    # C trajectory comparison with highlighted regions and zoomed views
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.1])
    ax_c_left = fig.add_subplot(gs[0, 0:2])
    ax_c_right = fig.add_subplot(gs[0, 2:4])
    ax_zoom_a = fig.add_subplot(gs[1, 0:2])
    ax_zoom_b = fig.add_subplot(gs[1, 2:4])

    _plot_trajectory(ax_c_left, case_map["C_medium_fixed_topo"], "C2 medium | M4 fixed-topo")
    _plot_trajectory(ax_c_right, case_map["C_medium_sp_smgf"], "C2 medium | SP-SMGF")
    ax_c_right.legend(loc="best", fontsize=8)

    zoom_a = ((-0.6, 4.8), (-2.2, 2.2))
    zoom_b = ((4.2, 8.8), (-1.8, 1.8))
    for ax in [ax_c_left, ax_c_right]:
        rect_a = Rectangle((zoom_a[0][0], zoom_a[1][0]), zoom_a[0][1] - zoom_a[0][0], zoom_a[1][1] - zoom_a[1][0], fill=False, linestyle='--', linewidth=1.2, edgecolor='tab:red')
        rect_b = Rectangle((zoom_b[0][0], zoom_b[1][0]), zoom_b[0][1] - zoom_b[0][0], zoom_b[1][1] - zoom_b[1][0], fill=False, linestyle='--', linewidth=1.2, edgecolor='tab:purple')
        ax.add_patch(rect_a)
        ax.add_patch(rect_b)
        ax.text(zoom_a[0][0], zoom_a[1][1] + 0.12, 'A', color='tab:red', fontsize=10, fontweight='bold')
        ax.text(zoom_b[0][0], zoom_b[1][1] + 0.12, 'B', color='tab:purple', fontsize=10, fontweight='bold')

    _plot_zoomed_trajectory(ax_zoom_a, case_map["C_medium_sp_smgf"], zoom_a[0], zoom_a[1], "Zoom A: corridor entry")
    _plot_zoomed_trajectory(ax_zoom_b, case_map["C_medium_sp_smgf"], zoom_b[0], zoom_b[1], "Zoom B: high-pressure corridor segment")
    fig.tight_layout()
    fig.savefig(output_root / "C_medium_trajectory_compare.png", dpi=200)
    plt.close(fig)

    # C state curves comparison
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex='col')
    _plot_state_curves(axes[:, 0], case_map["C_medium_fixed_topo"], "C2 medium states | M4")
    _plot_state_curves(axes[:, 1], case_map["C_medium_sp_smgf"], "C2 medium states | SP-SMGF")
    fig.tight_layout()
    fig.savefig(output_root / "C_medium_state_compare.png", dpi=200)
    plt.close(fig)

    # D1 trajectory sample
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_trajectory(ax, case_map["D1_prediction_trend"], "D1 high-speed tracking sample")
    fig.tight_layout()
    fig.savefig(output_root / "D1_prediction_sample.png", dpi=200)
    plt.close(fig)

    # E1 boundary trajectory
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_trajectory(ax, case_map["E1_boundary_case"], "E1 integrated challenge boundary case")
    fig.tight_layout()
    fig.savefig(output_root / "E1_boundary_case.png", dpi=200)
    plt.close(fig)

    _plot_c_group_bar(repo_root / "outputs", output_root)
    _plot_d1_trend(repo_root / "outputs", output_root)
