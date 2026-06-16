from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smgf.understanding_analysis import run_understanding_analysis


class UnderstandingAnalysisSmokeTests(unittest.TestCase):
    def test_run_understanding_analysis_writes_summary_and_trial_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "understanding"
            detail_df, summary_df = run_understanding_analysis(
                output_dir=output_dir,
                scenes=["c_geo_directional_passage_lite"],
                methods=["M28"],
                trials=1,
                seed_start=0,
            )

            self.assertFalse(detail_df.empty)
            self.assertFalse(summary_df.empty)
            self.assertTrue((output_dir / "understanding_trial_summary.csv").exists())
            self.assertTrue((output_dir / "understanding_summary.csv").exists())
            self.assertTrue((output_dir / "understanding_summary.json").exists())
            self.assertTrue((output_dir / "manifest.json").exists())

            trial_dir = output_dir / "trials" / "c_geo_directional_passage_lite" / "M28" / "seed_0"
            self.assertTrue((trial_dir / "agent_understanding.csv").exists())
            self.assertTrue((trial_dir / "global_understanding.csv").exists())
            self.assertTrue((trial_dir / "understanding_summary.json").exists())

            expected_summary_columns = {
                "corridor_activation_mean",
                "queue_mode_mean",
                "env_pressure_mean",
                "dominant_mode_switches_mean",
                "late_guard_scale_mean",
                "entry_flux_peak_mean",
                "risk_load_peak_mean",
                "unified_r_peak_mean",
            }
            self.assertTrue(expected_summary_columns.issubset(summary_df.columns))


if __name__ == "__main__":
    unittest.main()
