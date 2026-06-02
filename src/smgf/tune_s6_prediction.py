from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from smgf.experiments import METHOD_LIBRARY, SCENARIOS, run_scene_trial


def main() -> None:
    scene_key = "s6_fast_target"
    scene = SCENARIOS[scene_key]
    output_dir = Path("outputs/tune_s6_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    t_preds = [0.0, 0.3, 0.7, 1.0, 1.3, 1.5]
    beta_leads = [0.2, 0.35, 0.5]
    r_cs = [2.8, 3.2, 3.6]
    trials = 10

    rows = []

    for t_pred in t_preds:
        for beta_lead in beta_leads:
            for r_c in r_cs:
                params = replace(
                    scene.params,
                    t_pred=t_pred,
                    beta_lead=beta_lead,
                    r_c=r_c,
                )

                for method_key in ["M4", "M5", "M6", "M7"]:
                    for seed in range(trials):
                        result = run_scene_trial(
                            scene,
                            METHOD_LIBRARY[method_key],
                            seed=seed,
                            params_override=params,
                        )
                        row = asdict(result["metrics"])
                        row.update({
                            "scene": scene_key,
                            "method": method_key,
                            "seed": seed,
                            "t_pred": t_pred,
                            "beta_lead": beta_lead,
                            "r_c": r_c,
                        })
                        rows.append(row)

                print("finished", t_pred, beta_lead, r_c)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tune_s6_trials.csv", index=False)

    summary = (
        df.groupby(["method", "t_pred", "beta_lead", "r_c"])
        .agg(
            success_rate=("success", "mean"),
            collision_rate=("collisions", "mean"),
            completion_time_mean=("completion_time", "mean"),
            current_center_error_mean=("current_center_error", "mean"),
            predicted_center_error_mean=("predicted_center_error", "mean"),
            input_sat_mean=("input_saturation_ratio", "mean"),
            gmax_mean=("max_angle_gap_deg", "mean"),
            gmax_reach_time_mean=("gmax_reach_time", "mean"),
        )
        .reset_index()
        .sort_values(
            ["success_rate", "collision_rate", "current_center_error_mean"],
            ascending=[False, True, True],
        )
    )

    summary.to_csv(output_dir / "tune_s6_summary.csv", index=False)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
