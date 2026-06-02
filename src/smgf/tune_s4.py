from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from smgf.experiments import METHOD_LIBRARY, SCENARIOS, run_scene_trial


def _run_trial(task: tuple[dict[str, float], int, str, str]) -> dict[str, Any]:
    param_update, seed, scene_key, method_key = task
    scene = SCENARIOS[scene_key]
    params = replace(scene.params, **param_update)
    result = run_scene_trial(
        scene,
        METHOD_LIBRARY[method_key],
        seed=seed,
        params_override=params,
    )
    row = asdict(result["metrics"])
    row.update(param_update)
    row.update({"scene": scene_key, "method": method_key, "seed": seed})
    return row


def _summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(keys)
        .agg(
            success_rate=("success", "mean"),
            collision_rate=("collisions", "mean"),
            completion_time_mean=("completion_time", "mean"),
            min_obs_distance_mean=("min_obs_distance", "mean"),
            min_agent_distance_mean=("min_agent_distance", "mean"),
            gmax_mean=("max_angle_gap_deg", "mean"),
            radius_var_mean=("radius_variance", "mean"),
            input_sat_mean=("input_saturation_ratio", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            stall_steps_mean=("stall_steps", "mean"),
            inside_any_rate=("inside_any", "mean"),
            inside_final_rate=("inside_final", "mean"),
            gmax_success_rate=("gmax_success", "mean"),
            radius_success_rate=("radius_success", "mean"),
            sigma_success_rate=("sigma_success", "mean"),
            no_collision_rate=("no_collision", "mean"),
            dwell_success_rate=("dwell_success", "mean"),
            success_geom_final_rate=("success_geom_final", "mean"),
            gmax_reach_time_mean=("gmax_reach_time", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(
        by=["success_rate", "collision_rate"],
        ascending=[False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the S4 narrow-passage parameter grid.")
    parser.add_argument("--trials", type=int, default=int(os.environ.get("SMGF_TUNE_S4_TRIALS", "10")))
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("SMGF_TUNE_S4_OUTPUT", "outputs/tune_s4")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SMGF_TUNE_S4_WORKERS", str(os.cpu_count() or 1))))
    args = parser.parse_args()

    scene_key = "s4_narrow_passage"
    method_key = "M7"
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "eta_min": [0.05, 0.1, 0.2],
        "sigma_omega": [3.0, 5.0, 8.0],
        "k_t": [0.6, 0.9, 1.2],
        "r0": [1.2, 1.5, 1.8],
        "lambda_curl": [0.3, 0.5, 0.7],
    }

    keys = list(grid.keys())
    param_updates = [dict(zip(keys, values, strict=True)) for values in itertools.product(*[grid[k] for k in keys])]
    tasks = [(param_update, seed, scene_key, method_key) for param_update in param_updates for seed in range(args.trials)]

    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            rows.append(_run_trial(task))
            print(f"finished {idx}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for idx, row in enumerate(executor.map(_run_trial, tasks), start=1):
                rows.append(row)
                print(f"finished {idx}/{len(tasks)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tune_s4_trials.csv", index=False)

    summary = _summarize(df, keys)
    summary.to_csv(output_dir / "tune_s4_summary.csv", index=False)

    with (output_dir / "best_s4_params.json").open("w", encoding="utf-8") as f:
        json.dump(summary.head(20).to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
