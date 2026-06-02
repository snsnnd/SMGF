from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import Method, Params, SMGFController, Scenario, default_methods
from .metrics import TrialMetrics, evaluate_trial
from .scenarios import build_scenarios


METHOD_LIBRARY = default_methods()
SCENARIOS = build_scenarios()

EXPERIMENT_GROUPS = {
    "A_basic_modules": {
        "title": "Group A Basic Module Verification",
        "entries": [
            {"scene": "a1_single_goal_reach", "methods": ["M1", "M2"]},
            {"scene": "s1_single_obstacle", "methods": ["M1", "M2", "M3"]},
            {"scene": "a3_multi_agent_safety_crossing", "methods": ["M7", "M10"]},
        ],
    },
    "B_encirclement_geometry": {
        "title": "Group B Encirclement Geometry Verification",
        "entries": [
            {"scene": "b1_static_uniform_encirclement", "methods": ["M4", "M7"]},
            {"scene": "s2_same_side_expansion", "methods": ["M4", "M5", "M6", "M7", "M9"]},
            {"scene": "b3_moving_target_open_encirclement", "methods": ["M4", "M7", "M9"]},
        ],
    },
    "C_pressure_passage": {
        "title": "Group C Pressure and Passage Verification",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M4", "M5", "M6", "M7"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M4", "M5", "M6", "M7"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M4", "M5", "M6", "M7"]},
        ],
    },
    "D_prediction_navigation": {
        "title": "Group D Predictive Navigation Verification",
        "entries": [
            {"scene": "s6_fast_target", "methods": ["M4", "M7", "M9"]},
            {"scene": "d2_fast_target_single_obstacle", "methods": ["M4", "M7", "M9"]},
        ],
    },
    "E_integrated_challenges": {
        "title": "Group E Integrated Challenge Verification",
        "entries": [
            {"scene": "e1_tracking_single_obstacle", "methods": ["M4", "M6", "M7", "M9"]},
            {"scene": "e2_tracking_sparse_obstacles", "methods": ["M4", "M6", "M7", "M9"]},
            {"scene": "s5_dense_tracking_easy", "methods": ["M4", "M6", "M7", "M9"]},
            {"scene": "s5_dense_tracking_medium", "methods": ["M4", "M6", "M7", "M9"]},
            {"scene": "s5_dense_tracking_hard", "methods": ["M4", "M6", "M7", "M9"]},
        ],
    },
    "F_safe_tradeoff": {
        "title": "Group F Safety-Force Tradeoff Verification",
        "entries": [
            {"scene": "a3_multi_agent_safety_crossing", "methods": ["M7", "M10"]},
            {"scene": "b3_moving_target_open_encirclement", "methods": ["M7", "M10"]},
            {"scene": "s4_narrow_passage_easy", "methods": ["M7", "M10"]},
            {"scene": "s6_fast_target", "methods": ["M7", "M10"]},
            {"scene": "d2_fast_target_single_obstacle", "methods": ["M7", "M10"]},
        ],
    },
}


def _apply_seed_jitter(scene: Scenario, seed: int) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    positions = scene.initial_positions + rng.normal(scale=0.08, size=scene.initial_positions.shape)
    obstacles = []
    for obs in scene.obstacles:
        jitter = rng.normal(scale=0.03, size=2)
        obstacles.append(type(obs)(center=obs.center + jitter, radius=obs.radius))
    return positions, obstacles


def run_scene_trial(scene: Scenario, method: Method, seed: int = 0, params_override: Params | None = None) -> dict:
    params = params_override or scene.params
    controller = SMGFController(params, method, scene.n_agents)
    positions, obstacles = _apply_seed_jitter(scene, seed)
    steps = int(params.horizon / params.dt) + 1
    omega = np.zeros(scene.n_agents)

    positions_hist = np.zeros((steps, scene.n_agents, 2))
    u_hist = np.zeros((steps, scene.n_agents, 2))
    omega_hist = np.zeros((steps, scene.n_agents))
    rho_hist = np.zeros((steps, scene.n_agents))
    psi_hist = np.zeros((steps, scene.n_agents))
    phi_hist = np.zeros((steps, scene.n_agents))
    target_hist = np.zeros((steps, 2))
    predicted_target_hist = np.zeros((steps, 2))

    for k in range(steps):
        t = k * params.dt
        target = scene.target_fn(t)
        computed = controller.compute(positions, target, omega, obstacles, t)
        positions_hist[k] = positions
        u_hist[k] = computed["u"]
        omega_hist[k] = computed["omega"]
        rho_hist[k] = computed["rho"]
        psi_hist[k] = computed["psi_tilde"]
        phi_hist[k] = computed["phi"]
        target_hist[k] = target.position
        predicted_target_hist[k] = computed["predicted_target"]
        omega = computed["omega"]
        if k < steps - 1:
            positions = positions + params.dt * (computed["env"] + computed["u"])

    metrics = evaluate_trial(
        scene,
        params,
        positions_hist,
        target_hist,
        predicted_target_hist,
        u_hist,
        params.dt,
        obstacles=obstacles,
    )
    return {
        "scene": scene,
        "method": method,
        "seed": seed,
        "params": params,
        "obstacles": obstacles,
        "positions_hist": positions_hist,
        "u_hist": u_hist,
        "omega_hist": omega_hist,
        "rho_hist": rho_hist,
        "psi_hist": psi_hist,
        "phi_hist": phi_hist,
        "target_hist": target_hist,
        "predicted_target_hist": predicted_target_hist,
        "metrics": metrics,
    }


def _plot_trial(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = result["scene"]
    method = result["method"]
    positions_hist = result["positions_hist"]
    target_hist = result["target_hist"]
    predicted_hist = result["predicted_target_hist"]
    omega_hist = result["omega_hist"]
    rho_hist = result["rho_hist"]
    psi_hist = result["psi_hist"]
    dt = result["params"].dt
    time = np.arange(len(positions_hist)) * dt

    fig, ax = plt.subplots(figsize=(8, 6))
    for obs in result["obstacles"]:
        circle = plt.Circle(obs.center, obs.radius, color="gray", alpha=0.3)
        ax.add_patch(circle)
    for i in range(positions_hist.shape[1]):
        ax.plot(positions_hist[:, i, 0], positions_hist[:, i, 1], linewidth=1.4, label=f"agent {i + 1}")
        ax.scatter(positions_hist[0, i, 0], positions_hist[0, i, 1], s=20, marker="o")
        ax.scatter(positions_hist[-1, i, 0], positions_hist[-1, i, 1], s=30, marker="x")
    ax.plot(target_hist[:, 0], target_hist[:, 1], "k--", linewidth=2, label="target")
    ax.plot(predicted_hist[:, 0], predicted_hist[:, 1], color="tab:red", linestyle=":", linewidth=1.5, label="predicted target")
    ax.set_title(f"{scene.title} | {method.name}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(time, np.mean(psi_hist, axis=1), label="mean tilde_psi")
    axes[0].fill_between(time, np.min(psi_hist, axis=1), np.max(psi_hist, axis=1), alpha=0.2)
    axes[0].set_ylabel("tilde_psi")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time, np.mean(omega_hist, axis=1), label="mean Omega", color="tab:orange")
    axes[1].fill_between(time, np.min(omega_hist, axis=1), np.max(omega_hist, axis=1), color="tab:orange", alpha=0.2)
    axes[1].set_ylabel("Omega")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(time, np.mean(rho_hist, axis=1), label="mean rho", color="tab:green")
    axes[2].fill_between(time, np.min(rho_hist, axis=1), np.max(rho_hist, axis=1), color="tab:green", alpha=0.2)
    axes[2].set_ylabel("rho")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.25)
    fig.suptitle(f"State Modulation | {scene.title} | {method.name}")
    fig.tight_layout()
    fig.savefig(output_dir / "state_curves.png", dpi=180)
    plt.close(fig)


def _metrics_to_row(scene_key: str, method_key: str, seed: int, metrics: TrialMetrics) -> dict:
    row = asdict(metrics)
    row.update({"scene": scene_key, "method": method_key, "seed": seed})
    return row


def _entry_trial_task(task: tuple[str, str, int]) -> dict:
    scene_key, method_key, seed = task
    scene = SCENARIOS[scene_key]
    metrics = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed)["metrics"]
    return _metrics_to_row(scene_key, method_key, seed, metrics)


def _summarize_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    return (
        detail_df.groupby(["scene", "method"])
        .agg(
            success_rate=("success", "mean"),
            completion_time_mean=("completion_time", "mean"),
            completion_time_std=("completion_time", "std"),
            collision_rate=("collisions", "mean"),
            obs_collision_rate=("obs_collision", "mean"),
            agent_collision_rate=("agent_collision", "mean"),
            min_obs_distance_mean=("min_obs_distance", "mean"),
            min_agent_distance_mean=("min_agent_distance", "mean"),
            gmax_mean=("max_angle_gap_deg", "mean"),
            radius_var_mean=("radius_variance", "mean"),
            control_cost_mean=("control_cost", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            input_sat_mean=("input_saturation_ratio", "mean"),
            stall_steps_mean=("stall_steps", "mean"),
            inside_any_rate=("inside_any", "mean"),
            inside_final_rate=("inside_final", "mean"),
            gmax_success_rate=("gmax_success", "mean"),
            radius_success_rate=("radius_success", "mean"),
            sigma_success_rate=("sigma_success", "mean"),
            no_collision_rate=("no_collision", "mean"),
            dwell_success_rate=("dwell_success", "mean"),
            dwell_geom_success_rate=("dwell_geom_success", "mean"),
            dwell_success_no_collision_rate=("dwell_success_no_collision", "mean"),
            radius_error_final_mean=("radius_error_final", "mean"),
            success_geom_final_rate=("success_geom_final", "mean"),
            success_no_collision_rate=("success_no_collision", "mean"),
            gmax_reach_time_mean=("gmax_reach_time", "mean"),
            time_to_inside_mean=("time_to_inside", "mean"),
            time_to_gmax_mean=("time_to_gmax", "mean"),
            time_to_radius_mean=("time_to_radius", "mean"),
            time_to_full_geom_mean=("time_to_full_geom", "mean"),
            success_hold_time_mean=("success_hold_time", "mean"),
            post_success_violation_count_mean=("post_success_violation_count", "mean"),
            last_10s_gmax_mean=("last_10s_gmax_mean", "mean"),
            last_10s_radius_error_mean=("last_10s_radius_error_mean", "mean"),
            last_10s_inside_rate_mean=("last_10s_inside_rate", "mean"),
        )
        .reset_index()
    )


def _run_entries(entries: list[dict], output_path: Path, trials: int, seed_start: int = 0, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    tasks: list[tuple[str, str, int]] = []
    for entry in entries:
        scene_key = entry["scene"]
        scene = SCENARIOS[scene_key]
        for method_key in entry["methods"]:
            if scene.n_agents == 1 and method_key not in {"M1", "M2", "M3"}:
                continue
            result = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed_start)
            plot_dir = output_path / "examples" / scene_key / method_key
            _plot_trial(result, plot_dir)
            for seed in range(seed_start, seed_start + trials):
                tasks.append((scene_key, method_key, seed))

    resolved_workers = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    if resolved_workers <= 1:
        for task in tasks:
            rows.append(_entry_trial_task(task))
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            for row in executor.map(_entry_trial_task, tasks):
                rows.append(row)

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(output_path / "trial_metrics.csv", index=False)
    summary = _summarize_detail(detail_df)
    summary.to_csv(output_path / "summary_metrics.csv", index=False)
    with (output_path / "summary_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)
    return detail_df, summary


def run_suite(output_dir: str | Path, trials: int = 10, scenes: list[str] | None = None, methods: list[str] | None = None, seed_start: int = 0, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_scenes = scenes or [
        "s1_single_obstacle",
        "s2_same_side_expansion",
        "s3_static_encirclement",
        "s4_narrow_passage_easy",
        "s4_narrow_passage_medium",
        "s4_narrow_passage_hard",
        "s5_tracking_stage0",
        "s5_tracking_stage1",
        "s5_tracking_stage2",
        "s6_fast_target",
    ]
    selected_methods = methods or ["M1", "M3", "M4", "M5", "M6", "M7"]
    entries = [{"scene": scene_key, "methods": selected_methods} for scene_key in selected_scenes]
    detail_df, summary = _run_entries(entries, output_path, trials=trials, seed_start=seed_start, workers=workers)

    for scene_key in selected_scenes:
        scene_df = summary[summary["scene"] == scene_key]
        if scene_df.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].bar(scene_df["method"], scene_df["success_rate"], color="tab:blue")
        axes[0].set_title(f"{scene_key} success")
        axes[0].set_ylim(0, 1.05)
        axes[1].bar(scene_df["method"], scene_df["completion_time_mean"], color="tab:orange")
        axes[1].set_title("completion time")
        axes[2].bar(scene_df["method"], scene_df["gmax_mean"], color="tab:green")
        axes[2].set_title("max angle gap [deg]")
        for ax in axes:
            ax.grid(True, alpha=0.2)
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_path / f"{scene_key}_summary.png", dpi=180)
        plt.close(fig)

    return detail_df, summary


def run_group(group_key: str, output_dir: str | Path, trials: int = 10, seed_start: int = 0, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = EXPERIMENT_GROUPS[group_key]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "group_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({"group_key": group_key, "title": group["title"], "entries": group["entries"], "trials": trials, "seed_start": seed_start, "workers": workers}, fh, ensure_ascii=False, indent=2)
    detail_df, summary = _run_entries(group["entries"], output_path, trials=trials, seed_start=seed_start, workers=workers)
    return detail_df, summary
