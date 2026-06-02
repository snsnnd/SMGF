from __future__ import annotations

import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from smgf.experiments import METHOD_LIBRARY, SCENARIOS, run_scene_trial


def main() -> None:
    scene_key = "s4_narrow_passage"
    method_key = "M7"
    scene = SCENARIOS[scene_key]

    output_dir = Path("outputs/tune_s4")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "eta_min": [0.05, 0.1, 0.2],
        "sigma_omega": [3.0, 5.0, 8.0],
        "k_t": [0.6, 0.9, 1.2],
        "r0": [1.2, 1.5, 1.8],
        "lambda_curl": [0.3, 0.5, 0.7],
    }

    trials = 10
    rows = []

    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        param_update = dict(zip(keys, values, strict=True))
        params = replace(scene.params, **param_update)

        for seed in range(trials):
            result = run_scene_trial(
                scene,
                METHOD_LIBRARY[method_key],
                seed=seed,
                params_override=params,
            )
            row = asdict(result["metrics"])
            row.update(param_update)
            row.update({"scene": scene_key, "method": method_key, "seed": seed})
            rows.append(row)

        print("finished", param_update)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tune_s4_trials.csv", index=False)

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
            success_geom_final_rate=("success_geom_final", "mean"),
            gmax_reach_time_mean=("gmax_reach_time", "mean"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        by=["success_rate", "collision_rate", "completion_time_mean", "input_sat_mean", "control_smoothness_mean"],
        ascending=[False, True, True, True, True],
    )

    summary.to_csv(output_dir / "tune_s4_summary.csv", index=False)

    with (output_dir / "best_s4_params.json").open("w", encoding="utf-8") as f:
        json.dump(summary.head(20).to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
