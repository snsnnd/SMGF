from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_summary_tables(root: Path, output_file: Path) -> pd.DataFrame:
    summary_paths = sorted(root.glob("**/summary_metrics.csv"))
    rows = []
    for path in summary_paths:
        if path.resolve() == output_file.resolve():
            continue
        df = pd.read_csv(path)
        df.insert(0, "source_run", str(path.parent.relative_to(root.parent if output_file.is_absolute() else Path("."))))
        df.insert(1, "source_file", str(path))
        rows.append(df)

    if rows:
        merged = pd.concat(rows, ignore_index=True, sort=False)
        sort_cols = [col for col in ["source_run", "scene", "method"] if col in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)
    else:
        merged = pd.DataFrame(columns=["source_run", "source_file"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    return merged
