# SMGF Experiments

This repository implements a reproducible Python simulation framework for
State-Modulated Guidance Field (SMGF) experiments.

## Quick Start

```bash
/home/aaa/.local/bin/uv sync
/home/aaa/.local/bin/uv run smgf run-group --group A_basic_modules --trials 3 --output outputs/group_A_demo
```

## Restructured Experiment Layout

The experiment structure is no longer based on a single mixed `S1-S6` suite.
It is now organized into five layered groups:

1. `A_basic_modules`
2. `B_encirclement_geometry`
3. `C_pressure_passage`
4. `D_prediction_navigation`
5. `E_integrated_challenges`

This separates:

1. single-module validation
2. multi-module coupling validation
3. mechanism ablation
4. difficulty escalation
5. final integrated challenge scenes

## Main Commands

Run one scene-method pair:

```bash
/home/aaa/.local/bin/uv run smgf run-scene --scene s4_narrow_passage_medium --method M7 --seed 1 --output outputs/s4_demo
```

Run a restructured experiment group:

```bash
/home/aaa/.local/bin/uv run smgf run-group --group C_pressure_passage --trials 5 --seed-start 0 --output outputs/group_C
```

Run the legacy flat suite if needed:

```bash
/home/aaa/.local/bin/uv run smgf run-suite --trials 5 --seed-start 0 --output outputs/legacy_suite
```

## Seed Splits

Recommended protocol:

1. tuning seeds: `0-9`
2. validation seeds: `10-39`
3. final test seeds: `40-99`

Example:

```bash
/home/aaa/.local/bin/uv run smgf run-group --group C_pressure_passage --trials 10 --seed-start 0 --output outputs/tuning_C
/home/aaa/.local/bin/uv run smgf run-group --group C_pressure_passage --trials 30 --seed-start 10 --output outputs/validation_C
/home/aaa/.local/bin/uv run smgf run-group --group C_pressure_passage --trials 60 --seed-start 40 --output outputs/final_C
```

## Outputs

Generated outputs include:

1. trajectory plots
2. state-modulation plots
3. per-trial metrics
4. aggregated summary tables
5. group metadata for the restructured experiment sets
