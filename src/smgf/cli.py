from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .core import default_methods
from .experiments import EXPERIMENT_GROUPS, METHOD_LIBRARY, SCENARIOS, _plot_trial, run_group, run_scene_trial, run_suite
from .merge_results import merge_summary_tables
from .phase1 import SEED_SPLITS, export_phase1_plot_bundle, run_phase1_bundle, run_prediction_scan
from .plot_phase1 import plot_phase1_figures
from .trace_compare import compare_scene_traces, export_scene_trace
from .understanding_analysis import export_understanding_trial, run_understanding_analysis


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

    export_cmd = sub.add_parser("export-phase1-data", help="Export a curated phase-1 plotting bundle for MATLAB")
    export_cmd.add_argument("--output", type=str, default="data/phase1")

    plot_cmd = sub.add_parser("plot-phase1-figures", help="Render curated phase-1 paper figures with Python")
    plot_cmd.add_argument("--data-root", type=str, default="data/phase1")
    plot_cmd.add_argument("--output", type=str, default="figures/phase1")

    trace_cmd = sub.add_parser("export-trace", help="Export full trace tables for one scene/method/seed")
    trace_cmd.add_argument("--scene", choices=sorted(SCENARIOS.keys()), required=True)
    trace_cmd.add_argument("--method", choices=sorted(METHOD_LIBRARY.keys()), required=True)
    trace_cmd.add_argument("--seed", type=int, default=0)
    trace_cmd.add_argument("--t-pred", type=float, default=None)
    trace_cmd.add_argument("--output", type=str, required=True)

    compare_cmd = sub.add_parser("compare-traces", help="Run two methods on the same seed and export divergence traces")
    compare_cmd.add_argument("--scene", choices=sorted(SCENARIOS.keys()), required=True)
    compare_cmd.add_argument("--method-a", choices=sorted(METHOD_LIBRARY.keys()), required=True)
    compare_cmd.add_argument("--method-b", choices=sorted(METHOD_LIBRARY.keys()), required=True)
    compare_cmd.add_argument("--seed", type=int, default=0)
    compare_cmd.add_argument("--t-pred", type=float, default=None)
    compare_cmd.add_argument("--output", type=str, required=True)

    understanding_trial_cmd = sub.add_parser("export-understanding-trial", help="Export environment-understanding traces for one scene/method/seed")
    understanding_trial_cmd.add_argument("--scene", choices=sorted(SCENARIOS.keys()), required=True)
    understanding_trial_cmd.add_argument("--method", choices=sorted(METHOD_LIBRARY.keys()), required=True)
    understanding_trial_cmd.add_argument("--seed", type=int, default=0)
    understanding_trial_cmd.add_argument("--t-pred", type=float, default=None)
    understanding_trial_cmd.add_argument("--output", type=str, required=True)

    understanding_cmd = sub.add_parser("run-understanding-analysis", help="Run the environment-understanding analysis bundle")
    understanding_cmd.add_argument("--scenes", nargs="*", choices=sorted(SCENARIOS.keys()), default=None)
    understanding_cmd.add_argument("--methods", nargs="*", choices=sorted(METHOD_LIBRARY.keys()), default=None)
    understanding_cmd.add_argument("--trials", type=int, default=5)
    understanding_cmd.add_argument("--seed-start", type=int, default=0)
    understanding_cmd.add_argument("--t-pred", type=float, default=None)
    understanding_cmd.add_argument("--output", type=str, required=True)

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
    elif args.command == "export-phase1-data":
        produced = export_phase1_plot_bundle(Path(args.output))
        print(json.dumps(produced, ensure_ascii=False, indent=2))
    elif args.command == "plot-phase1-figures":
        plot_phase1_figures(Path.cwd(), Path(args.data_root), Path(args.output))
        print(json.dumps({"data_root": args.data_root, "output": args.output}, ensure_ascii=False, indent=2))
    elif args.command == "export-trace":
        payload = export_scene_trace(args.scene, args.method, args.seed, Path(args.output), t_pred=args.t_pred)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "compare-traces":
        payload = compare_scene_traces(args.scene, args.method_a, args.method_b, args.seed, Path(args.output), t_pred=args.t_pred)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "export-understanding-trial":
        payload = export_understanding_trial(args.scene, args.method, args.seed, Path(args.output), t_pred=args.t_pred)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "run-understanding-analysis":
        _, summary = run_understanding_analysis(
            Path(args.output),
            scenes=args.scenes,
            methods=args.methods,
            trials=args.trials,
            seed_start=args.seed_start,
            t_pred=args.t_pred,
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
