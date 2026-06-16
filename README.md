# SMGF Experiments

This repository implements a reproducible Python simulation framework for
State-Modulated Guidance Field (SMGF) experiments.

## Repository Guide

Current high-value entry points:

1. `SMGF研究规划文档.md`
   Current project positioning, method rationale, and future research branches.
2. `docs/exploration/M43-M62方法分析与M43_M62交叉验证_20260616.md`
   Current `M43-M62` corridor/energy/release branch analysis and latest validation results.
3. `docs/phase1/README.md`
   Phase-1 experiment and paper-oriented documentation index.
4. `data/phase1/` and `figures/phase1/`
   Curated plotting data and generated paper figures.
5. `paper/phase1/`
   Frontiers template and current manuscript drafts.

Historical reports and older experiment snapshots have been archived or superseded by the documentation under `docs/` and `versions/`.

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

Run the phase-1 bundle with seed splits:

```bash
/home/aaa/.local/bin/uv run smgf run-phase1 --split tuning --output outputs/phase1_tuning
/home/aaa/.local/bin/uv run smgf run-phase1 --split validation --output outputs/phase1_validation
/home/aaa/.local/bin/uv run smgf run-phase1 --split final --output outputs/phase1_final
```

Run a predictive navigation scan only:

```bash
/home/aaa/.local/bin/uv run smgf run-prediction-scan --scene s6_fast_target --split tuning --output outputs/d1_scan
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

Active branch-exploration outputs currently worth reading include:

1. `outputs/bc_m43_m62_cross_type_validation_v1/`
2. `outputs/bd_m43_m62_dense_tracking_hard_v1/`
3. `outputs/bb_m62_extended_corridor_probe_v1/`

## Documentation Policy

1. Keep active methodology, experiment, and paper-writing documents under `docs/`.
2. Keep historical or superseded materials under `versions/`.
3. Avoid storing outdated standalone reports in the repository root once they have been superseded.
