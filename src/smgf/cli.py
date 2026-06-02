from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .core import default_methods
from .experiments import EXPERIMENT_GROUPS, METHOD_LIBRARY, SCENARIOS, _plot_trial, run_group, run_scene_trial, run_suite
from .merge_results import merge_summary_tables
from .phase1 import SEED_SPLITS, run_phase1_bundle, run_prediction_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SMGF experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    scene_cmd = sub.add_parser("run-scene", help="Run one scenario/method pair")
    scene_cmd.add_argument("--scene", choices=sorted(SCENARIOS.keys()), required=True)
    scene_cmd.add_argument("--method", choices=sorted(METHOD_LIBRARY.keys()), required=True)
    scene_cmd.add_argument("--seed", type=int, default=0)
    scene_cmd.add_argument("--output", type=str, required=True)
    scene_cmd.add_argument("--t-pred", type=float, default=None, help="Override predictive horizon for the selected run")

    suite_cmd = sub.add_parser("run-suite", help="Run the main experiment suite")
    suite_cmd.add_argument("--trials", type=int, default=10)
    suite_cmd.add_argument("--output", type=str, required=True)
    suite_cmd.add_argument("--seed-start", type=int, default=0)
    suite_cmd.add_argument("--workers", type=int, default=None)

    group_cmd = sub.add_parser("run-group", help="Run a restructured experiment group")
    group_cmd.add_argument("--group", choices=sorted(EXPERIMENT_GROUPS.keys()), required=True)
    group_cmd.add_argument("--trials", type=int, default=10)
    group_cmd.add_argument("--seed-start", type=int, default=0)
    group_cmd.add_argument("--workers", type=int, default=None)
    group_cmd.add_argument("--output", type=str, required=True)

    merge_cmd = sub.add_parser("merge-results", help="Merge summary tables into one table")
    merge_cmd.add_argument("--root", type=str, default="outputs")
    merge_cmd.add_argument("--output", type=str, default="outputs/merged_summary_metrics.csv")

    pred_cmd = sub.add_parser("run-prediction-scan", help="Run D1 or D2 predictive navigation scan")
    pred_cmd.add_argument("--scene", choices=["s6_fast_target", "d2_fast_target_single_obstacle"], required=True)
    pred_cmd.add_argument("--split", choices=sorted(SEED_SPLITS.keys()), default="tuning")
    pred_cmd.add_argument("--trials", type=int, default=None)
    pred_cmd.add_argument("--seed-start", type=int, default=None)
    pred_cmd.add_argument("--workers", type=int, default=None)
    pred_cmd.add_argument("--output", type=str, required=True)

    phase1_cmd = sub.add_parser("run-phase1", help="Run the Phase-1 experiment bundle with seed splits")
    phase1_cmd.add_argument("--split", choices=sorted(SEED_SPLITS.keys()), default="tuning")
    phase1_cmd.add_argument("--trials", type=int, default=None)
    phase1_cmd.add_argument("--seed-start", type=int, default=None)
    phase1_cmd.add_argument("--group-workers", type=int, default=None)
    phase1_cmd.add_argument("--prediction-workers", type=int, default=None)
    phase1_cmd.add_argument("--output", type=str, required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run-scene":
        scene = SCENARIOS[args.scene]
        params = scene.params if args.t_pred is None else replace(scene.params, t_pred=args.t_pred)
        result = run_scene_trial(scene, METHOD_LIBRARY[args.method], seed=args.seed, params_override=params)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_trial(result, output_dir)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(result["metrics"]), fh, ensure_ascii=False, indent=2)
        print(json.dumps(asdict(result["metrics"]), ensure_ascii=False, indent=2))
    elif args.command == "run-suite":
        _, summary = run_suite(args.output, trials=args.trials, seed_start=args.seed_start, workers=args.workers)
        print(summary.to_string(index=False))
    elif args.command == "run-group":
        _, summary = run_group(args.group, args.output, trials=args.trials, seed_start=args.seed_start, workers=args.workers)
        print(summary.to_string(index=False))
    elif args.command == "merge-results":
        merged = merge_summary_tables(Path(args.root), Path(args.output))
        print(merged.to_string(index=False))
    elif args.command == "run-prediction-scan":
        _, summary = run_prediction_scan(args.scene, Path(args.output), split=args.split, trials=args.trials, seed_start=args.seed_start, workers=args.workers)
        print(summary.to_string(index=False))
    elif args.command == "run-phase1":
        produced = run_phase1_bundle(Path(args.output), split=args.split, trials=args.trials, seed_start=args.seed_start, group_workers=args.group_workers, prediction_workers=args.prediction_workers)
        print(json.dumps(produced, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
