from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .experiments import EXPERIMENT_GROUPS, METHOD_LIBRARY, SCENARIOS, _metrics_to_row, _summarize_detail, run_group, run_scene_trial


SEED_SPLITS = {
    "tuning": {"seed_start": 0, "trials": 10},
    "validation": {"seed_start": 10, "trials": 30},
    "final": {"seed_start": 40, "trials": 60},
}


PHASE1_PLOT_BUNDLE = [
    {
        "figure_key": "A2_obstacle_avoidance",
        "group": "A",
        "scene": "s1_single_obstacle",
        "method": "M3",
        "seed": 10,
        "notes": "Single-obstacle avoidance positive example for the basic module section.",
        "params_override": {},
    },
    {
        "figure_key": "B3_open_encirclement_m7",
        "group": "B",
        "scene": "b3_moving_target_open_encirclement",
        "method": "M7",
        "seed": 10,
        "notes": "Open-space moving-target encirclement example with the main SMGF method.",
        "params_override": {},
    },
    {
        "figure_key": "B3_open_encirclement_m9",
        "group": "B",
        "scene": "b3_moving_target_open_encirclement",
        "method": "M9",
        "seed": 10,
        "notes": "Open-space moving-target encirclement example with angle regularization.",
        "params_override": {},
    },
    {
        "figure_key": "C_medium_fixed_topo",
        "group": "C",
        "scene": "s4_narrow_passage_medium",
        "method": "M4",
        "seed": 10,
        "notes": "Medium corridor fixed-topology reference trajectory.",
        "params_override": {},
    },
    {
        "figure_key": "C_medium_original_m7",
        "group": "C",
        "scene": "s4_narrow_passage_medium",
        "method": "M7",
        "seed": 10,
        "notes": "Medium corridor original SMGF failure-mode trajectory.",
        "params_override": {},
    },
    {
        "figure_key": "C_medium_sp_smgf",
        "group": "C",
        "scene": "s4_narrow_passage_medium",
        "method": "M7",
        "seed": 37,
        "notes": "Medium corridor structure-preserving SMGF candidate trajectory.",
        "params_override": {
            "k_s": 3.4,
            "k_t": 1.5,
            "r0": 1.9,
            "topo_rho_floor": 0.65,
            "topo_floor_on_threshold": 0.0,
            "topo_floor_off_threshold": 0.0,
            "topo_floor_release_tau": 1.0,
        },
    },
    {
        "figure_key": "D1_prediction_trend",
        "group": "D1",
        "scene": "s6_fast_target",
        "method": "M7",
        "seed": 10,
        "notes": "High-speed target tracking trend example used to illustrate D1 behavior.",
        "params_override": {"t_pred": 1.0},
    },
    {
        "figure_key": "E1_boundary_case",
        "group": "E",
        "scene": "e1_tracking_single_obstacle",
        "method": "M7",
        "seed": 10,
        "notes": "Integrated challenge boundary example showing obstacle-tracking coupling failure.",
        "params_override": {},
    },
]


def _params_to_json_dict(params) -> dict:
    payload = asdict(params)
    env_flow = payload.get("env_flow")
    if callable(env_flow):
        payload["env_flow"] = getattr(env_flow, "__name__", str(env_flow))
    return payload


def _prediction_scan_task(task: tuple[str, str, float, int]) -> dict:
    scene_key, method_key, t_pred, seed = task
    scene = SCENARIOS[scene_key]
    params = replace(scene.params, t_pred=t_pred)
    result = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed, params_override=params)
    row = _metrics_to_row(scene_key, method_key, seed, result["metrics"])
    row.update({"t_pred": t_pred})
    return row


def _resolve_split(split: str, trials: int | None, seed_start: int | None) -> tuple[int, int]:
    config = SEED_SPLITS[split]
    return (seed_start if seed_start is not None else config["seed_start"], trials if trials is not None else config["trials"])


def run_prediction_scan(
    scene_key: str,
    output_dir: Path,
    split: str,
    trials: int | None = None,
    seed_start: int | None = None,
    workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, count = _resolve_split(split, trials, seed_start)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_preds = [0.0, 0.3, 0.7, 1.0, 1.3, 1.5]
    tasks = [
        (scene_key, method_key, t_pred, seed)
        for t_pred in t_preds
        for method_key in ["M4", "M7", "M9"]
        for seed in range(start, start + count)
    ]
    resolved_workers = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    rows = []
    if resolved_workers <= 1:
        for task in tasks:
            row = _prediction_scan_task(task)
            row.update({"split": split})
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            for row in executor.map(_prediction_scan_task, tasks):
                row.update({"split": split})
                rows.append(row)
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "trial_metrics.csv", index=False)
    summary = (
        detail.groupby(["scene", "method", "t_pred", "split"])
        .agg(
            success_rate=("success", "mean"),
            collision_rate=("collisions", "mean"),
            obs_collision_rate=("obs_collision", "mean"),
            agent_collision_rate=("agent_collision", "mean"),
            completion_time_mean=("completion_time", "mean"),
            current_center_error_mean=("current_center_error", "mean"),
            predicted_center_error_mean=("predicted_center_error", "mean"),
            input_sat_mean=("input_saturation_ratio", "mean"),
            gmax_mean=("max_angle_gap_deg", "mean"),
            radius_error_final_mean=("radius_error_final", "mean"),
            time_to_gmax_mean=("time_to_gmax", "mean"),
            time_to_radius_mean=("time_to_radius", "mean"),
        )
        .reset_index()
        .sort_values(["method", "t_pred"])
    )
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)
    return detail, summary


def run_phase1_bundle(
    output_root: Path,
    split: str,
    trials: int | None = None,
    seed_start: int | None = None,
    group_workers: int | None = None,
    prediction_workers: int | None = None,
) -> dict[str, str]:
    start, count = _resolve_split(split, trials, seed_start)
    output_root.mkdir(parents=True, exist_ok=True)
    produced: dict[str, str] = {}

    for group_key in ["A_basic_modules", "B_encirclement_geometry", "C_pressure_passage"]:
        group_dir = output_root / group_key
        run_group(group_key, group_dir, trials=count, seed_start=start, workers=group_workers)
        produced[group_key] = str(group_dir)

    d1_dir = output_root / "D1_prediction_scan"
    run_prediction_scan("s6_fast_target", d1_dir, split=split, trials=count, seed_start=start, workers=prediction_workers)
    produced["D1_prediction_scan"] = str(d1_dir)

    d2_dir = output_root / "D2_prediction_scan"
    run_prediction_scan("d2_fast_target_single_obstacle", d2_dir, split=split, trials=count, seed_start=start, workers=prediction_workers)
    produced["D2_prediction_scan"] = str(d2_dir)

    with (output_root / "phase1_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump({"split": split, "seed_start": start, "trials": count, "outputs": produced}, fh, ensure_ascii=False, indent=2)

    return produced


def _write_plot_bundle_entry(output_dir: Path, entry: dict) -> dict[str, str | int | dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = SCENARIOS[entry["scene"]]
    params = replace(scene.params, **entry["params_override"]) if entry["params_override"] else scene.params
    result = run_scene_trial(scene, METHOD_LIBRARY[entry["method"]], seed=entry["seed"], params_override=params)

    positions_hist = result["positions_hist"]
    u_hist = result["u_hist"]
    omega_hist = result["omega_hist"]
    rho_hist = result["rho_hist"]
    psi_hist = result["psi_hist"]
    phi_hist = result["phi_hist"]
    target_hist = result["target_hist"]
    predicted_target_hist = result["predicted_target_hist"]
    dt = result["params"].dt
    time = [step * dt for step in range(len(positions_hist))]

    trajectory_rows: list[dict] = []
    for step_idx, t in enumerate(time):
        for agent_idx in range(positions_hist.shape[1]):
            trajectory_rows.append(
                {
                    "step": step_idx,
                    "time": t,
                    "agent": agent_idx,
                    "x": float(positions_hist[step_idx, agent_idx, 0]),
                    "y": float(positions_hist[step_idx, agent_idx, 1]),
                    "ux": float(u_hist[step_idx, agent_idx, 0]),
                    "uy": float(u_hist[step_idx, agent_idx, 1]),
                }
            )

    state_rows: list[dict] = []
    for step_idx, t in enumerate(time):
        for agent_idx in range(omega_hist.shape[1]):
            state_rows.append(
                {
                    "step": step_idx,
                    "time": t,
                    "agent": agent_idx,
                    "omega": float(omega_hist[step_idx, agent_idx]),
                    "rho": float(rho_hist[step_idx, agent_idx]),
                    "psi_tilde": float(psi_hist[step_idx, agent_idx]),
                    "phi": float(phi_hist[step_idx, agent_idx]),
                }
            )

    target_rows = [
        {
            "step": step_idx,
            "time": t,
            "target_x": float(target_hist[step_idx, 0]),
            "target_y": float(target_hist[step_idx, 1]),
            "pred_target_x": float(predicted_target_hist[step_idx, 0]),
            "pred_target_y": float(predicted_target_hist[step_idx, 1]),
        }
        for step_idx, t in enumerate(time)
    ]

    obstacle_rows = [
        {
            "obstacle": obs_idx,
            "center_x": float(obs.center[0]),
            "center_y": float(obs.center[1]),
            "radius": float(obs.radius),
        }
        for obs_idx, obs in enumerate(result["obstacles"])
    ]

    metrics_dict = asdict(result["metrics"])
    recommended_end_time = result["params"].horizon
    if metrics_dict["dwell_success"]:
        recommended_end_time = min(result["params"].horizon, metrics_dict["completion_time"] + 5.0)
    elif metrics_dict["collisions"]:
        collision_rows = []
        for step_idx, t in enumerate(time):
            step_positions = positions_hist[step_idx]
            min_agent_distance = float("inf")
            for i in range(len(step_positions)):
                for j in range(i + 1, len(step_positions)):
                    dij = float(((step_positions[i] - step_positions[j]) ** 2).sum() ** 0.5)
                    min_agent_distance = min(min_agent_distance, dij)
            if min_agent_distance < result["params"].d_agent_safe:
                collision_rows.append(t)
        if collision_rows:
            recommended_end_time = min(result["params"].horizon, collision_rows[0] + 5.0)

    pd.DataFrame(trajectory_rows).to_csv(output_dir / "trajectory.csv", index=False)
    pd.DataFrame(state_rows).to_csv(output_dir / "state_curves.csv", index=False)
    pd.DataFrame(target_rows).to_csv(output_dir / "target_curves.csv", index=False)
    pd.DataFrame(obstacle_rows, columns=["obstacle", "center_x", "center_y", "radius"]).to_csv(output_dir / "obstacles.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_dict, fh, ensure_ascii=False, indent=2)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "figure_key": entry["figure_key"],
                "group": entry["group"],
                "scene": entry["scene"],
                "method": entry["method"],
                "seed": entry["seed"],
                "notes": entry["notes"],
                "params": _params_to_json_dict(result["params"]),
                "params_override": entry["params_override"],
                "recommended_plot_end_time": recommended_end_time,
                "horizon": result["params"].horizon,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "figure_key": entry["figure_key"],
        "group": entry["group"],
        "scene": entry["scene"],
        "method": entry["method"],
        "seed": entry["seed"],
        "path": str(output_dir),
    }


def export_phase1_plot_bundle(output_root: Path) -> list[dict[str, str | int | dict]]:
    output_root.mkdir(parents=True, exist_ok=True)
    exported = []
    for entry in PHASE1_PLOT_BUNDLE:
        figure_dir = output_root / entry["group"] / entry["figure_key"]
        exported.append(_write_plot_bundle_entry(figure_dir, entry))

    with (output_root / "bundle_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "description": "Phase-1 MATLAB plotting bundle for paper figures.",
                "entries": exported,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    return exported
