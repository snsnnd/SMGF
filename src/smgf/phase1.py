from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .experiments import EXPERIMENT_GROUPS, METHOD_LIBRARY, SCENARIOS, _metrics_to_row, _summarize_detail, run_group, run_scene_trial


SEED_SPLITS = {
    "tuning": {"seed_start": 0, "trials": 10},
    "validation": {"seed_start": 10, "trials": 30},
    "final": {"seed_start": 40, "trials": 60},
}


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
