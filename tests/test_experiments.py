from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smgf.experiments import run_suite


class RunSuiteSmokeTests(unittest.TestCase):
    def test_run_suite_writes_metrics_with_failure_decomposition_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "suite"
            detail_df, summary_df = run_suite(
                output_dir=output_dir,
                trials=1,
                scenes=["s1_single_obstacle"],
                methods=["M1"],
            )

            self.assertFalse(detail_df.empty)
            self.assertFalse(summary_df.empty)
            self.assertTrue((output_dir / "trial_metrics.csv").exists())
            self.assertTrue((output_dir / "summary_metrics.csv").exists())
            self.assertTrue((output_dir / "summary_metrics.json").exists())

            expected_columns = {
                "obs_collision_rate",
                "agent_collision_rate",
                "inside_any_rate",
                "inside_final_rate",
                "gmax_success_rate",
                "radius_success_rate",
                "sigma_success_rate",
                "no_collision_rate",
                "dwell_success_rate",
                "dwell_success_no_collision_rate",
                "time_to_full_geom_mean",
            }
            self.assertTrue(expected_columns.issubset(summary_df.columns))


if __name__ == "__main__":
    unittest.main()
