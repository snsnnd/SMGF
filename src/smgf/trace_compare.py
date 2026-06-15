from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import METHOD_LIBRARY, SCENARIOS, run_scene_trial


DECISION_COLUMNS = ["progress", "caution", "reshape", "spacing"]
SCALE_COLUMNS = ["nav_scale", "safe_scale", "directional_scale", "guard_scale", "enc_scale", "angle_scale"]


def _result_time_vector(result: dict) -> np.ndarray:
    dt = result["params"].dt
    return np.arange(len(result["positions_hist"])) * dt


def _agent_trace_dataframe(result: dict) -> pd.DataFrame:
    time = _result_time_vector(result)
    positions_hist = result["positions_hist"]
    u_hist = result["u_hist"]
    omega_hist = result["omega_hist"]
    rho_hist = result["rho_hist"]
    topo_gain_hist = result["topo_gain_hist"]
    psi_hist = result["psi_hist"]
    phi_hist = result["phi_hist"]
    xi_axis_hist = result.get("xi_axis_hist")
    decision_z_hist = result.get("decision_z_hist")
    coverage_aux_hist = result.get("coverage_aux_hist")
    nav_scale_hist = result.get("nav_scale_hist")
    safe_scale_hist = result.get("safe_scale_hist")
    directional_scale_hist = result.get("directional_scale_hist")
    guard_scale_hist = result.get("guard_scale_hist")
    enc_scale_hist = result.get("enc_scale_hist")
    angle_scale_hist = result.get("angle_scale_hist")
    target_hist = result["target_hist"]
    predicted_target_hist = result["predicted_target_hist"]

    rows: list[dict] = []
    for step_idx, t in enumerate(time):
        for agent_idx in range(positions_hist.shape[1]):
            row = {
                "step": step_idx,
                "time": float(t),
                "agent": agent_idx,
                "x": float(positions_hist[step_idx, agent_idx, 0]),
                "y": float(positions_hist[step_idx, agent_idx, 1]),
                "ux": float(u_hist[step_idx, agent_idx, 0]),
                "uy": float(u_hist[step_idx, agent_idx, 1]),
                "omega": float(omega_hist[step_idx, agent_idx]),
                "rho": float(rho_hist[step_idx, agent_idx]),
                "topo_gain": float(topo_gain_hist[step_idx, agent_idx]),
                "psi_tilde": float(psi_hist[step_idx, agent_idx]),
                "phi": float(phi_hist[step_idx, agent_idx]),
                "target_x": float(target_hist[step_idx, 0]),
                "target_y": float(target_hist[step_idx, 1]),
                "pred_target_x": float(predicted_target_hist[step_idx, 0]),
                "pred_target_y": float(predicted_target_hist[step_idx, 1]),
                "xi_axis_x": float(xi_axis_hist[step_idx, agent_idx, 0]) if xi_axis_hist is not None else 0.0,
                "xi_axis_y": float(xi_axis_hist[step_idx, agent_idx, 1]) if xi_axis_hist is not None else 0.0,
                "coverage_aux": float(coverage_aux_hist[step_idx, agent_idx]) if coverage_aux_hist is not None else 0.0,
                "nav_scale": float(nav_scale_hist[step_idx, agent_idx]) if nav_scale_hist is not None else 1.0,
                "safe_scale": float(safe_scale_hist[step_idx, agent_idx]) if safe_scale_hist is not None else 1.0,
                "directional_scale": float(directional_scale_hist[step_idx, agent_idx]) if directional_scale_hist is not None else 1.0,
                "guard_scale": float(guard_scale_hist[step_idx, agent_idx]) if guard_scale_hist is not None else 1.0,
                "enc_scale": float(enc_scale_hist[step_idx, agent_idx]) if enc_scale_hist is not None else 1.0,
                "angle_scale": float(angle_scale_hist[step_idx, agent_idx]) if angle_scale_hist is not None else 1.0,
            }
            if decision_z_hist is not None:
                for z_idx, label in enumerate(DECISION_COLUMNS):
                    row[label] = float(decision_z_hist[step_idx, agent_idx, z_idx])
            else:
                for label in DECISION_COLUMNS:
                    row[label] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _write_trace_bundle(result: dict, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_df = _agent_trace_dataframe(result)
    trace_df.to_csv(output_dir / "agent_trace.csv", index=False)

    obstacle_rows = [
        {
            "obstacle": obs_idx,
            "center_x": float(obs.center[0]),
            "center_y": float(obs.center[1]),
            "radius": float(obs.radius),
        }
        for obs_idx, obs in enumerate(result["obstacles"])
    ]
    pd.DataFrame(obstacle_rows, columns=["obstacle", "center_x", "center_y", "radius"]).to_csv(output_dir / "obstacles.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(result["metrics"]), fh, ensure_ascii=False, indent=2)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "scene": result["scene"].key,
                "method": result["method"].name,
                "seed": result["seed"],
                "params": asdict(result["params"]),
            },
            fh,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return trace_df


def _first_persistent_time(mask: np.ndarray, times: np.ndarray, run_len: int) -> float | None:
    streak = 0
    for idx, is_true in enumerate(mask):
        if is_true:
            streak += 1
            if streak >= run_len:
                return float(times[idx - run_len + 1])
        else:
            streak = 0
    return None


def _build_comparison_timeseries(trace_a: pd.DataFrame, trace_b: pd.DataFrame) -> pd.DataFrame:
    merged = trace_a.merge(trace_b, on=["step", "time", "agent"], suffixes=("_a", "_b"))
    angle_a = np.arctan2(merged["xi_axis_y_a"], merged["xi_axis_x_a"])
    angle_b = np.arctan2(merged["xi_axis_y_b"], merged["xi_axis_x_b"])
    axis_diff = np.abs((angle_a - angle_b + np.pi) % (2 * np.pi) - np.pi)
    merged["xi_axis_diff_deg"] = np.degrees(axis_diff)
    merged["position_diff"] = np.sqrt((merged["x_a"] - merged["x_b"]) ** 2 + (merged["y_a"] - merged["y_b"]) ** 2)
    merged["u_diff"] = np.sqrt((merged["ux_a"] - merged["ux_b"]) ** 2 + (merged["uy_a"] - merged["uy_b"]) ** 2)
    merged["z_l2_diff"] = np.sqrt(sum((merged[f"{name}_a"] - merged[f"{name}_b"]) ** 2 for name in DECISION_COLUMNS))
    merged["scale_l2_diff"] = np.sqrt(sum((merged[f"{name}_a"] - merged[f"{name}_b"]) ** 2 for name in SCALE_COLUMNS))

    agg_spec = {
        "xi_axis_diff_deg": "mean",
        "position_diff": "mean",
        "u_diff": "mean",
        "z_l2_diff": "mean",
        "scale_l2_diff": "mean",
        "x_a": "mean",
        "y_a": "mean",
        "x_b": "mean",
        "y_b": "mean",
    }
    for name in DECISION_COLUMNS + SCALE_COLUMNS:
        agg_spec[f"{name}_a"] = "mean"
        agg_spec[f"{name}_b"] = "mean"
    per_time = merged.groupby(["step", "time"]).agg(agg_spec).reset_index()
    per_time["center_distance"] = np.sqrt((per_time["x_a"] - per_time["x_b"]) ** 2 + (per_time["y_a"] - per_time["y_b"]) ** 2)
    return per_time


def _divergence_summary(per_time: pd.DataFrame) -> dict:
    times = per_time["time"].to_numpy()
    decision_time = _first_persistent_time((per_time["z_l2_diff"] > 0.12).to_numpy(), times, run_len=5)
    axis_time = _first_persistent_time((per_time["xi_axis_diff_deg"] > 15.0).to_numpy(), times, run_len=5)
    geometry_time = _first_persistent_time((per_time["center_distance"] > 0.2).to_numpy(), times, run_len=5)
    control_time = _first_persistent_time((per_time["scale_l2_diff"] > 0.18).to_numpy(), times, run_len=5)
    candidates = [value for value in [decision_time, axis_time, control_time, geometry_time] if value is not None]
    primary_time = min(candidates) if candidates else None
    return {
        "thresholds": {
            "z_l2_diff": 0.12,
            "xi_axis_diff_deg": 15.0,
            "center_distance": 0.2,
            "scale_l2_diff": 0.18,
            "run_len": 5,
        },
        "first_decision_divergence_time": decision_time,
        "first_axis_divergence_time": axis_time,
        "first_control_divergence_time": control_time,
        "first_geometry_divergence_time": geometry_time,
        "primary_divergence_time": primary_time,
        "peak_diffs": {
            "z_l2_diff_max": float(per_time["z_l2_diff"].max()),
            "xi_axis_diff_deg_max": float(per_time["xi_axis_diff_deg"].max()),
            "scale_l2_diff_max": float(per_time["scale_l2_diff"].max()),
            "center_distance_max": float(per_time["center_distance"].max()),
        },
    }


def export_scene_trace(scene_key: str, method_key: str, seed: int, output_dir: Path, t_pred: float | None = None) -> dict:
    scene = SCENARIOS[scene_key]
    params = replace(scene.params, t_pred=t_pred) if t_pred is not None else scene.params
    result = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed, params_override=params)
    _write_trace_bundle(result, output_dir)
    return {
        "scene": scene_key,
        "method": method_key,
        "seed": seed,
        "output": str(output_dir),
    }


def compare_scene_traces(scene_key: str, method_a: str, method_b: str, seed: int, output_dir: Path, t_pred: float | None = None) -> dict:
    scene = SCENARIOS[scene_key]
    params = replace(scene.params, t_pred=t_pred) if t_pred is not None else scene.params
    result_a = run_scene_trial(scene, METHOD_LIBRARY[method_a], seed=seed, params_override=params)
    result_b = run_scene_trial(scene, METHOD_LIBRARY[method_b], seed=seed, params_override=params)

    trace_a = _write_trace_bundle(result_a, output_dir / method_a)
    trace_b = _write_trace_bundle(result_b, output_dir / method_b)
    per_time = _build_comparison_timeseries(trace_a, trace_b)
    per_time.to_csv(output_dir / "comparison_timeseries.csv", index=False)

    summary = {
        "scene": scene_key,
        "seed": seed,
        "method_a": method_a,
        "method_b": method_b,
        **_divergence_summary(per_time),
    }
    with (output_dir / "divergence_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary
