from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .core import default_methods
from .experiments import METHOD_LIBRARY, SCENARIOS, _plot_trial, run_scene_trial, run_suite


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
        _, summary = run_suite(args.output, trials=args.trials)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
