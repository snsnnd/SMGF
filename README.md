# SMGF Experiments

This project implements a reproducible Python simulation framework for the
State-Modulated Guidance Field (SMGF) experiments described in the paper notes.

## Quick start

```bash
/home/aaa/.local/bin/uv sync
/home/aaa/.local/bin/uv run smgf run-suite --trials 5 --output outputs/quick
```

## Main commands

```bash
/home/aaa/.local/bin/uv run smgf run-scene --scene s4_narrow_passage --method M7 --seed 1 --output outputs/s4_demo
/home/aaa/.local/bin/uv run smgf run-suite --trials 10 --output outputs/suite
```

The generated outputs include trajectory plots, state-modulation plots, per-trial
metrics, and aggregated summary tables.
