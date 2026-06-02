from __future__ import annotations

import json
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


def run_suite(output_dir: str | Path, trials: int = 10, scenes: list[str] | None = None, methods: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_scenes = scenes or [
        "s1_single_obstacle",
        "s2_same_side_expansion",
        "s3_static_encirclement",
        "s4_narrow_passage",
        "s5_dense_tracking_easy",
        "s5_dense_tracking_medium",
        "s5_dense_tracking_hard",
        "s6_fast_target",
    ]
    selected_methods = methods or ["M1", "M3", "M4", "M5", "M6", "M7"]
    rows = []

    for scene_key in selected_scenes:
        scene = SCENARIOS[scene_key]
        for method_key in selected_methods:
            if scene.n_agents == 1 and method_key not in {"M1", "M2", "M3"}:
                continue
            result = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=0)
            plot_dir = output_path / "examples" / scene_key / method_key
            _plot_trial(result, plot_dir)
            for seed in range(trials):
                metrics = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed)["metrics"]
                rows.append(_metrics_to_row(scene_key, method_key, seed, metrics))

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(output_path / "trial_metrics.csv", index=False)

    summary = (
        detail_df.groupby(["scene", "method"])
        .agg(
            success_rate=("success", "mean"),
            completion_time_mean=("completion_time", "mean"),
            completion_time_std=("completion_time", "std"),
            collision_rate=("collisions", "mean"),
            min_obs_distance_mean=("min_obs_distance", "mean"),
            min_agent_distance_mean=("min_agent_distance", "mean"),
            gmax_mean=("max_angle_gap_deg", "mean"),
            radius_var_mean=("radius_variance", "mean"),
            control_cost_mean=("control_cost", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            input_sat_mean=("input_saturation_ratio", "mean"),
            stall_steps_mean=("stall_steps", "mean"),
            inside_final_rate=("inside_final", "mean"),
            radius_error_final_mean=("radius_error_final", "mean"),
            success_geom_final_rate=("success_geom_final", "mean"),
            success_no_collision_rate=("success_no_collision", "mean"),
            gmax_reach_time_mean=("gmax_reach_time", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_path / "summary_metrics.csv", index=False)

    with (output_path / "summary_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)

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
