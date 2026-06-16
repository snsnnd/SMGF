from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


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


def _plot_trajectory(ax, case: dict, title: str | None = None, show_final_ring: bool = False) -> None:
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
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def _plot_zoomed_trajectory(ax, case: dict, xlim: tuple[float, float], ylim: tuple[float, float], title: str | None = None) -> None:
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
    ax.grid(True, alpha=0.2)


def _min_agent_distance_series(case: dict) -> pd.DataFrame:
    metadata = case["metadata"]
    trajectory = _truncate_by_time(case["trajectory"], metadata["recommended_plot_end_time"])
    rows = []
    for step, group in trajectory.groupby("step"):
        coords = group[["x", "y"]].to_numpy()
        min_dist = float("inf")
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dij = float(np.linalg.norm(coords[i] - coords[j]))
                min_dist = min(min_dist, dij)
        rows.append({"step": step, "time": float(group.iloc[0]["time"]), "min_agent_distance": min_dist})
    return pd.DataFrame(rows)


def _plot_min_distance_time_series(output_root: Path, case_map: dict) -> None:
    mapping = [
        ("Fixed-topology", case_map["C_medium_fixed_topo"], "tab:blue"),
        ("SMGF-Core", case_map["C_medium_original_m7"], "tab:orange"),
        ("SP-SMGF-C", case_map["C_medium_sp_smgf"], "tab:green"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, case, color in mapping:
        series = _min_agent_distance_series(case)
        ax.plot(series["time"], series["min_agent_distance"], label=label, color=color, linewidth=1.6)
    d_safe = case_map["C_medium_sp_smgf"]["metadata"]["params"]["d_agent_safe"]
    ax.axhline(d_safe, color="tab:red", linestyle="--", linewidth=1.2, label="d_agent_safe")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("min agent distance")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / "C_medium_min_distance_timeseries.png", dpi=200)
    fig.savefig(output_root / "C_medium_min_distance_timeseries_clean.png", dpi=200)
    plt.close(fig)


def _plot_snapshot(ax, case: dict, snapshot_time: float, title: str | None = None) -> None:
    trajectory = case["trajectory"]
    target = case["target"]
    obstacles = case["obstacles"]
    snapshot = trajectory.iloc[(trajectory["time"] - snapshot_time).abs().argsort()].groupby("agent").head(1).sort_values("agent")
    target_row = target.iloc[(target["time"] - snapshot_time).abs().argsort()].iloc[0]

    for _, obs in obstacles.iterrows():
        circle = plt.Circle((obs["center_x"], obs["center_y"]), obs["radius"], color="gray", alpha=0.25)
        ax.add_patch(circle)

    ax.scatter(snapshot["x"], snapshot["y"], s=38, color="tab:blue", zorder=3)
    for _, row in snapshot.iterrows():
        ax.text(row["x"], row["y"] + 0.08, f"{int(row['agent'])}", fontsize=8, ha="center")

    center = np.array([target_row["target_x"], target_row["target_y"]], dtype=float)
    snapshot = snapshot.copy()
    snapshot["angle"] = np.mod(np.arctan2(snapshot["y"] - center[1], snapshot["x"] - center[0]), 2 * np.pi)
    snapshot = snapshot.sort_values("angle")
    ring_x = snapshot["x"].tolist() + [snapshot.iloc[0]["x"]]
    ring_y = snapshot["y"].tolist() + [snapshot.iloc[0]["y"]]
    ax.plot(ring_x, ring_y, color="tab:purple", linestyle="-.", linewidth=1.4)

    ax.scatter([target_row["target_x"]], [target_row["target_y"]], color="black", s=50, marker="*", zorder=4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def _plot_c_snapshot_comparison(output_root: Path, case_map: dict) -> None:
    original_state = case_map["C_medium_original_m7"]["state"]
    mean_omega = original_state.groupby("time")["omega"].mean().reset_index()
    snapshot_time = float(mean_omega.loc[mean_omega["omega"].idxmax(), "time"])
    mapping = [
        ("Fixed-topology baseline", case_map["C_medium_fixed_topo"]),
        ("SMGF-Core", case_map["C_medium_original_m7"]),
        ("SP-SMGF-C", case_map["C_medium_sp_smgf"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (label, case) in zip(axes, mapping):
        _plot_snapshot(ax, case, snapshot_time, f"{label} @ t={snapshot_time:.2f}s")
    fig.tight_layout()
    fig.savefig(output_root / "C_medium_snapshot_compare.png", dpi=200)
    plt.close(fig)


def _plot_state_curves(axs, case: dict, title: str | None = None) -> None:
    metadata = case["metadata"]
    state = _truncate_by_time(case["state"], metadata["recommended_plot_end_time"])
    mean_state = state.groupby("time").agg(
        omega_mean=("omega", "mean"),
        omega_min=("omega", "min"),
        omega_max=("omega", "max"),
        rho_mean=("rho", "mean"),
        rho_min=("rho", "min"),
        rho_max=("rho", "max"),
        topo_gain_mean=("topo_gain", "mean"),
        topo_gain_min=("topo_gain", "min"),
        topo_gain_max=("topo_gain", "max"),
        psi_mean=("psi_tilde", "mean"),
        psi_min=("psi_tilde", "min"),
        psi_max=("psi_tilde", "max"),
    ).reset_index()

    series = [
        ("omega", "omega_mean", "omega_min", "omega_max", "tab:orange"),
        ("rho", "rho_mean", "rho_min", "rho_max", "tab:green"),
        ("psi", "psi_mean", "psi_min", "psi_max", "tab:blue"),
    ]
    for ax, (label, mean_key, min_key, max_key, color) in zip(axs, series):
        ax.plot(mean_state["time"], mean_state[mean_key], color=color, linewidth=1.6, label=f"raw {label}" if label == "rho" else None)
        ax.fill_between(mean_state["time"], mean_state[min_key], mean_state[max_key], color=color, alpha=0.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)

    rho_ax = axs[1]
    rho_ax.plot(
        mean_state["time"],
        mean_state["topo_gain_mean"],
        color="tab:red",
        linewidth=2.2,
        linestyle=(0, (6, 3)),
        label="applied $\\rho^{eff}$",
        zorder=4,
    )
    topo_floor = float(case["metadata"]["params"].get("topo_rho_floor", 0.0))
    if topo_floor > 0.0:
        rho_ax.axhline(
            topo_floor,
            color="black",
            linewidth=1.8,
            linestyle=":",
            alpha=0.9,
            label="$\\rho_{floor}$",
            zorder=3,
        )
    rho_ax.legend(loc="best", fontsize=8)

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
    # Paper figures use in-panel labels rather than top titles.


def _plot_c_group_bar(outputs_root: Path, figure_dir: Path) -> None:
    summary = pd.read_csv(outputs_root / "c_group_candidate_full_compare_v2" / "summary_metrics.csv")
    summary = summary[summary["method"].isin(["M4", "M5", "M6", "M7"])]
    scenes = ["s4_narrow_passage_easy", "s4_narrow_passage_medium", "s4_narrow_passage_hard"]
    method_order = ["M4", "M5", "M6", "M7"]
    method_labels = {"M4": "Fixed-topology", "M5": "SMGF w/o Psi", "M6": "SMGF w/o Omega", "M7": "SMGF-Core"}
    labels = ["easy", "medium", "hard"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.18
    xs = range(len(scenes))
    for idx, method in enumerate(method_order):
        sub = summary[summary["method"] == method].set_index("scene")
        axes[0].bar([x + (idx - 1.5) * width for x in xs], [sub.loc[s, "success_rate"] for s in scenes], width=width, label=method_labels[method])
        axes[1].bar([x + (idx - 1.5) * width for x in xs], [sub.loc[s, "agent_collision_rate"] for s in scenes], width=width, label=method_labels[method])

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
    method_labels = {"M4": "Fixed-topology", "M7": "SMGF-Core", "M9": "SMGF-Angle"}
    method_styles = {
        "M4": {"color": "#1f5fd1", "linestyle": "--", "marker": "o", "markerfacecolor": "white", "markeredgewidth": 1.5, "zorder": 6, "linewidth": 2.4, "markersize": 7.0},
        "M7": {"color": "tab:orange", "linestyle": "-", "marker": "o", "markerfacecolor": "tab:orange", "markeredgewidth": 1.0, "zorder": 4},
        "M9": {"color": "tab:green", "linestyle": "-.", "marker": "^", "markerfacecolor": "tab:green", "markeredgewidth": 1.0, "zorder": 5},
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("completion_time_mean", "completion time"),
        ("current_center_error_mean", "current center error"),
        ("gmax_mean", "gmax [deg]"),
        ("radius_error_final_mean", "radius error"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for method, group in summary.groupby("method"):
            style = method_styles[method]
            ax.plot(
                group["t_pred"],
                group[metric],
                linewidth=style.get("linewidth", 1.7),
                label=method_labels[method],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=style.get("markersize", 5.5),
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=style["markeredgewidth"],
                zorder=style["zorder"],
            )
        ax.set_xlabel("T_p")
        ax.grid(True, alpha=0.2)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "D1_prediction_trends.png", dpi=200)
    plt.close(fig)


def _plot_d2_coupling(outputs_root: Path, figure_dir: Path) -> None:
    summary = pd.read_csv(outputs_root / "phase1_validation_v2" / "D2_prediction_scan" / "summary_metrics.csv")
    summary = summary[summary["method"].isin(["M4", "M7", "M9"])]
    method_labels = {"M4": "Fixed-topology", "M7": "SMGF-Core", "M9": "SMGF-Angle"}
    method_styles = {
        "M4": {"color": "#1f5fd1", "linestyle": "--", "marker": "o", "markerfacecolor": "white", "markeredgewidth": 1.5, "zorder": 6, "linewidth": 2.4, "markersize": 7.0},
        "M7": {"color": "tab:orange", "linestyle": "-", "marker": "o", "markerfacecolor": "tab:orange", "markeredgewidth": 1.0, "zorder": 4},
        "M9": {"color": "tab:green", "linestyle": "-.", "marker": "^", "markerfacecolor": "tab:green", "markeredgewidth": 1.0, "zorder": 5},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8))
    for method, group in summary.groupby("method"):
        group = group.sort_values("t_pred")
        style = method_styles[method]
        axes[0].plot(
            group["t_pred"],
            group["obs_collision_rate"],
            linewidth=style.get("linewidth", 1.8),
            label=method_labels[method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=style.get("markersize", 5.5),
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=style["markeredgewidth"],
            zorder=style["zorder"],
        )
        axes[1].plot(
            group["t_pred"],
            group["current_center_error_mean"],
            linewidth=style.get("linewidth", 1.8),
            label=method_labels[method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=style.get("markersize", 5.5),
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=style["markeredgewidth"],
            zorder=style["zorder"],
        )

    axes[0].set_xlabel("T_p")
    axes[0].set_ylabel("obstacle collision rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_xlabel("T_p")
    axes[1].set_ylabel("current center error")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(figure_dir / "D2_prediction_obstacle_coupling.png", dpi=200)
    plt.close(fig)


def _plot_failure_heatmap(outputs_root: Path, figure_dir: Path) -> None:
    sources = {
        "b": pd.read_csv(outputs_root / "phase1_final_v1" / "B_encirclement_geometry" / "summary_metrics.csv"),
        "c": pd.read_csv(outputs_root / "phase1_final_v1" / "C_pressure_passage" / "summary_metrics.csv"),
        "d1": pd.read_csv(outputs_root / "phase1_final_v1" / "D1_prediction_scan" / "summary_metrics.csv"),
        "g": pd.read_csv(outputs_root / "group_G_external_final_v1" / "summary_metrics.csv"),
        "e": pd.read_csv(outputs_root / "group_E_final_v1" / "summary_metrics.csv"),
    }

    def _pick(df: pd.DataFrame, **filters) -> pd.Series:
        mask = np.ones(len(df), dtype=bool)
        for key, value in filters.items():
            mask &= df[key] == value
        picked = df.loc[mask]
        if picked.empty:
            raise ValueError(f"No row found for filters {filters}")
        return picked.iloc[0]

    regimes = [
        ("B1 geometry-only", _pick(sources["b"], scene="b1_static_uniform_encirclement", method="M7")),
        ("C2 compression-dominant", _pick(sources["c"], scene="s4_narrow_passage_medium", method="M7")),
        ("D1 pursuit-compression", _pick(sources["d1"], scene="s6_fast_target", method="M7", t_pred=1.5)),
        ("D2 obstacle-coupled", _pick(sources["g"], scene="d2_fast_target_single_obstacle", method="M4")),
        ("E1 mixed-failure", _pick(sources["e"], scene="e1_tracking_single_obstacle", method="M7")),
    ]

    def _geometry_score(row: pd.Series) -> float:
        no_collision = float(row.get("no_collision_rate", 1.0 - row.get("collision_rate", 1.0)))
        success = float(row.get("success_rate", 0.0))
        if success > 0.0 or no_collision < 0.9:
            return 0.0
        gmax_fail = 1.0 - float(row.get("gmax_success_rate", 1.0))
        radius_fail = 1.0 - float(row.get("radius_success_rate", 1.0))
        return max(gmax_fail, radius_fail)

    def _dominant_label(geom: float, agent: float, obstacle: float) -> str:
        if obstacle >= 0.5 and agent >= 0.5:
            return "mixed"
        if agent >= 0.5 and obstacle < 0.5:
            return "compression"
        if obstacle >= 0.5 and agent < 0.5:
            return "obstacle"
        if geom >= 0.5:
            return "geometry"
        return "weak / mixed"

    row_labels = []
    data = []
    dominant_labels = []
    for label, row in regimes:
        geom = _geometry_score(row)
        agent = float(row.get("agent_collision_rate", 0.0))
        obstacle = float(row.get("obs_collision_rate", 0.0))
        mixed = min(agent, obstacle)
        row_labels.append(label)
        data.append([geom, agent, obstacle, mixed])
        dominant_labels.append(_dominant_label(geom, agent, obstacle))

    data_arr = np.array(data, dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "failure_heat",
        ["#fcfcfd", "#f1f4f8", "#e2e9f0", "#cad7e4", "#a9bfd4"],
    )

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    im = ax.imshow(data_arr, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(
        range(4),
        [
            "geometry\ndeficit",
            "inter-agent\ncompression",
            "obstacle\ninteraction",
            "coupled obstacle-agent\npressure",
        ],
    )
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[1]):
            value = data_arr[i, j]
            text_color = "#2b2f2c"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5, color=text_color)
        ax.text(4.08, i, dominant_labels[i], ha="left", va="center", fontsize=9, fontweight="bold", color="#36413c")

    ax.text(4.08, -0.78, "dominant reading", ha="left", va="bottom", fontsize=9, color="#36413c")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.03)
    cbar.set_label("failure intensity / rate", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(figure_dir / "failure_mode_dominant_matrix.png", dpi=200)
    fig.savefig(figure_dir / "failure_mode_dominant_matrix_clean.png", dpi=200)
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
    _plot_trajectory(axes[0], case_map["B3_open_encirclement_m7"], show_final_ring=True)
    _plot_trajectory(axes[1], case_map["B3_open_encirclement_m9"], show_final_ring=True)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / "B3_open_encirclement_compare.png", dpi=200)
    fig.savefig(output_root / "B3_open_encirclement_compare_clean.png", dpi=200)
    plt.close(fig)

    # C trajectory comparison with highlighted regions and zoomed views
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 6, height_ratios=[2.0, 1.1])
    ax_c_left = fig.add_subplot(gs[0, 0:2])
    ax_c_mid = fig.add_subplot(gs[0, 2:4])
    ax_c_right = fig.add_subplot(gs[0, 4:6])
    ax_zoom_a = fig.add_subplot(gs[1, 0:3])
    ax_zoom_b = fig.add_subplot(gs[1, 3:6])

    _plot_trajectory(ax_c_left, case_map["C_medium_fixed_topo"])
    _plot_trajectory(ax_c_mid, case_map["C_medium_original_m7"])
    _plot_trajectory(ax_c_right, case_map["C_medium_sp_smgf"])
    ax_c_right.legend(loc="best", fontsize=8)
    for ax, panel_label in zip([ax_c_left, ax_c_mid, ax_c_right], ["Fixed-topology", "SMGF-Core", "SP-SMGF-C"]):
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="0.7"))

    zoom_a = ((-0.6, 4.8), (-2.2, 2.2))
    zoom_b = ((4.2, 8.8), (-1.8, 1.8))
    for ax in [ax_c_left, ax_c_mid, ax_c_right]:
        rect_a = Rectangle((zoom_a[0][0], zoom_a[1][0]), zoom_a[0][1] - zoom_a[0][0], zoom_a[1][1] - zoom_a[1][0], fill=False, linestyle='--', linewidth=1.2, edgecolor='tab:red')
        rect_b = Rectangle((zoom_b[0][0], zoom_b[1][0]), zoom_b[0][1] - zoom_b[0][0], zoom_b[1][1] - zoom_b[1][0], fill=False, linestyle='--', linewidth=1.2, edgecolor='tab:purple')
        ax.add_patch(rect_a)
        ax.add_patch(rect_b)
        ax.text(zoom_a[0][0], zoom_a[1][1] + 0.12, 'A', color='tab:red', fontsize=10, fontweight='bold')
        ax.text(zoom_b[0][0], zoom_b[1][1] + 0.12, 'B', color='tab:purple', fontsize=10, fontweight='bold')

    _plot_zoomed_trajectory(ax_zoom_a, case_map["C_medium_original_m7"], zoom_a[0], zoom_a[1])
    _plot_zoomed_trajectory(ax_zoom_b, case_map["C_medium_sp_smgf"], zoom_b[0], zoom_b[1])
    ax_zoom_a.text(0.02, 0.98, "A: SMGF-Core at corridor entry", transform=ax_zoom_a.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="0.7"))
    ax_zoom_b.text(0.02, 0.98, "B: SP-SMGF-C in high-pressure segment", transform=ax_zoom_b.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="0.7"))
    fig.tight_layout()
    fig.savefig(output_root / "C_medium_trajectory_compare.png", dpi=200)
    fig.savefig(output_root / "C_medium_trajectory_compare_clean.png", dpi=200)
    plt.close(fig)

    # C state curves comparison
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex='col')
    _plot_state_curves(axes[:, 0], case_map["C_medium_fixed_topo"])
    _plot_state_curves(axes[:, 1], case_map["C_medium_sp_smgf"])
    axes[0, 0].text(0.02, 0.98, "Fixed-topology", transform=axes[0, 0].transAxes, ha="left", va="top", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="0.7"))
    axes[0, 1].text(0.02, 0.98, "SP-SMGF-C", transform=axes[0, 1].transAxes, ha="left", va="top", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="0.7"))
    fig.tight_layout()
    fig.savefig(output_root / "C_medium_state_compare.png", dpi=200)
    fig.savefig(output_root / "C_medium_state_compare_clean.png", dpi=200)
    plt.close(fig)

    # D1 trajectory sample
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_trajectory(ax, case_map["D1_prediction_trend"])
    fig.tight_layout()
    fig.savefig(output_root / "D1_prediction_sample.png", dpi=200)
    plt.close(fig)

    # E1 boundary trajectory
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_trajectory(ax, case_map["E1_boundary_case"])
    fig.tight_layout()
    fig.savefig(output_root / "E1_boundary_case.png", dpi=200)
    plt.close(fig)

    _plot_c_group_bar(repo_root / "outputs", output_root)
    _plot_d1_trend(repo_root / "outputs", output_root)
    _plot_d2_coupling(repo_root / "outputs", output_root)
    _plot_failure_heatmap(repo_root / "outputs", output_root)
    _plot_min_distance_time_series(output_root, case_map)
    _plot_c_snapshot_comparison(output_root, case_map)

    # Keep the manuscript's clean overview figure filenames synchronized.
    overview_map = {
        "C_medium_trajectory_compare.png": "C_medium_trajectory_compare_clean.png",
        "C_medium_state_compare.png": "C_medium_state_compare_clean.png",
        "C_medium_min_distance_timeseries.png": "C_medium_min_distance_timeseries_clean.png",
        "B3_open_encirclement_compare.png": "B3_open_encirclement_compare_clean.png",
    }
    for src_name, dst_name in overview_map.items():
        src = output_root / src_name
        dst = output_root / dst_name
        if src.exists() and src != dst:
            shutil.copyfile(src, dst)
