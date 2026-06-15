from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smgf.experiments import EXPERIMENT_GROUPS, METHOD_LIBRARY, SCENARIOS, run_group, run_suite
from smgf.phase1 import run_phase1_bundle, run_prediction_scan


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
                "lateral_width_final_mean",
                "longitudinal_span_final_mean",
                "queue_stability_mean",
                "gmax_success_rate",
                "radius_success_rate",
                "sigma_success_rate",
                "no_collision_rate",
                "dwell_success_rate",
                "dwell_success_no_collision_rate",
                "time_to_full_geom_mean",
            }
            self.assertTrue(expected_columns.issubset(summary_df.columns))

    def test_run_group_writes_group_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "group"
            detail_df, summary_df = run_group(
                group_key="A_basic_modules",
                output_dir=output_dir,
                trials=1,
                seed_start=0,
            )

            self.assertFalse(detail_df.empty)
            self.assertFalse(summary_df.empty)
            self.assertTrue((output_dir / "group_metadata.json").exists())
            self.assertIn("A_basic_modules", EXPERIMENT_GROUPS)

    def test_run_prediction_scan_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "prediction"
            detail_df, summary_df = run_prediction_scan(
                scene_key="s6_fast_target",
                output_dir=output_dir,
                split="tuning",
                trials=1,
                seed_start=0,
            )
            self.assertFalse(detail_df.empty)
            self.assertFalse(summary_df.empty)
            self.assertTrue((output_dir / "summary_metrics.csv").exists())

    def test_geo_lite_method_and_scenes_are_registered(self) -> None:
        self.assertIn("M13", METHOD_LIBRARY)
        self.assertIn("M14", METHOD_LIBRARY)
        self.assertIn("M15", METHOD_LIBRARY)
        self.assertIn("M16", METHOD_LIBRARY)
        self.assertIn("M17", METHOD_LIBRARY)
        self.assertIn("M18", METHOD_LIBRARY)
        self.assertIn("M19", METHOD_LIBRARY)
        self.assertIn("M20", METHOD_LIBRARY)
        self.assertIn("M21", METHOD_LIBRARY)
        self.assertIn("M22", METHOD_LIBRARY)
        self.assertIn("M23", METHOD_LIBRARY)
        self.assertIn("M24", METHOD_LIBRARY)
        self.assertIn("M25", METHOD_LIBRARY)
        self.assertIn("M26", METHOD_LIBRARY)
        self.assertIn("M27", METHOD_LIBRARY)
        self.assertIn("M28", METHOD_LIBRARY)
        self.assertIn("M29", METHOD_LIBRARY)
        self.assertIn("M30", METHOD_LIBRARY)
        self.assertIn("M31", METHOD_LIBRARY)
        self.assertIn("M32", METHOD_LIBRARY)
        self.assertIn("M33", METHOD_LIBRARY)
        self.assertIn("M34", METHOD_LIBRARY)
        self.assertIn("M35", METHOD_LIBRARY)
        self.assertIn("M36", METHOD_LIBRARY)
        self.assertIn("M37", METHOD_LIBRARY)
        self.assertIn("M38", METHOD_LIBRARY)
        self.assertIn("M39", METHOD_LIBRARY)
        self.assertIn("M40", METHOD_LIBRARY)
        self.assertIn("M41", METHOD_LIBRARY)
        self.assertIn("M42", METHOD_LIBRARY)
        self.assertIn("M43", METHOD_LIBRARY)
        self.assertIn("M46", METHOD_LIBRARY)
        self.assertIn("M47", METHOD_LIBRARY)
        self.assertIn("M48", METHOD_LIBRARY)
        self.assertIn("M49", METHOD_LIBRARY)
        self.assertIn("M50", METHOD_LIBRARY)
        self.assertIn("M51", METHOD_LIBRARY)
        self.assertIn("M52", METHOD_LIBRARY)
        self.assertIn("M53", METHOD_LIBRARY)
        self.assertIn("M54", METHOD_LIBRARY)
        self.assertIn("M55", METHOD_LIBRARY)
        self.assertIn("M56", METHOD_LIBRARY)
        self.assertIn("M57", METHOD_LIBRARY)
        self.assertIn("M58", METHOD_LIBRARY)
        self.assertIn("M59", METHOD_LIBRARY)
        self.assertIn("M60", METHOD_LIBRARY)
        self.assertIn("c_geo_directional_passage_lite", SCENARIOS)
        self.assertIn("d_geo_fast_target_lite", SCENARIOS)
        self.assertIn("e_geo_tracking_single_obstacle_lite", SCENARIOS)
        self.assertIn("H_geo_directional_lite", EXPERIMENT_GROUPS)
        self.assertIn("I_branch_basic_validation", EXPERIMENT_GROUPS)
        self.assertIn("J_hybrid_basic_validation", EXPERIMENT_GROUPS)
        self.assertIn("K_scheme_comparison_validation", EXPERIMENT_GROUPS)
        self.assertIn("L_fusion_basic_validation", EXPERIMENT_GROUPS)
        self.assertIn("M_tensor_basic_validation", EXPERIMENT_GROUPS)
        self.assertIn("N_tensor_energy_polar_validation", EXPERIMENT_GROUPS)
        self.assertIn("O_exploration_saved_comparison", EXPERIMENT_GROUPS)
        self.assertIn("P_continuous_polar_mode_validation", EXPERIMENT_GROUPS)
        self.assertIn("Q_xi_self_localization_validation", EXPERIMENT_GROUPS)
        self.assertIn("R_xi_on_m21_validation", EXPERIMENT_GROUPS)
        self.assertIn("S_orca_lite_comparison", EXPERIMENT_GROUPS)
        self.assertIn("T_cbf_qp_lite_comparison", EXPERIMENT_GROUPS)
        self.assertIn("U_mpc_cbf_lite_comparison", EXPERIMENT_GROUPS)
        self.assertIn("V_magnetic_corridor_validation", EXPERIMENT_GROUPS)
        self.assertIn("W_xi_magnetic_polarity_validation", EXPERIMENT_GROUPS)
        self.assertIn("X_pure_magnetic_corridor_validation", EXPERIMENT_GROUPS)
        self.assertIn("Y_temporal_magnetic_corridor_validation", EXPERIMENT_GROUPS)
        self.assertIn("Z_magnetic_energy_validation", EXPERIMENT_GROUPS)
        self.assertIn("AA_pure_soft_boundary_validation", EXPERIMENT_GROUPS)
        self.assertIn("AB_queue_occupancy_soft_validation", EXPERIMENT_GROUPS)
        self.assertIn("AC_rear_push_soft_validation", EXPERIMENT_GROUPS)
        self.assertIn("AD_xi_polarity_axis_validation", EXPERIMENT_GROUPS)
        self.assertIn("AE_xi_polarity_axis_high_energy_validation", EXPERIMENT_GROUPS)
        self.assertIn("AF_xi_polarity_axis_structured_validation", EXPERIMENT_GROUPS)
        self.assertIn("AG_xi_polarity_axis_high_energy_followups", EXPERIMENT_GROUPS)
        self.assertIn("AH_hard_energy_adaptation_validation", EXPERIMENT_GROUPS)
        self.assertIn("M44_m43_harder_validation", EXPERIMENT_GROUPS)
        self.assertIn("AJ_structured_energy_tank_validation", EXPERIMENT_GROUPS)
        self.assertIn("AK_dual_stage_energy_validation", EXPERIMENT_GROUPS)
        self.assertIn("AL_dual_stage_activation_validation", EXPERIMENT_GROUPS)
        self.assertIn("AM_factorized_energy_ablation_validation", EXPERIMENT_GROUPS)
        self.assertIn("AN_entry_energy_propagation_validation", EXPERIMENT_GROUPS)
        self.assertIn("AO_leader_driven_entry_validation", EXPERIMENT_GROUPS)
        self.assertIn("AP_global_quota_redistribution_validation", EXPERIMENT_GROUPS)
        self.assertIn("AQ_unified_state_entry_push_validation", EXPERIMENT_GROUPS)

    def test_run_phase1_bundle_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "phase1"
            produced = run_phase1_bundle(output_dir, split="tuning", trials=1, seed_start=0)
            self.assertIn("A_basic_modules", produced)
            self.assertTrue((output_dir / "phase1_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
