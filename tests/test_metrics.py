from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from smgf.core import Obstacle, Params, Scenario, Target
from smgf.metrics import evaluate_trial


class EvaluateTrialSmokeTests(unittest.TestCase):
    def _scenario(self) -> Scenario:
        params = replace(Params(), horizon=0.05, d_obs_safe=0.45, d_agent_safe=0.0, r_c=1.0)
        return Scenario(
            key="jitter_eval_test",
            title="Jitter Evaluation Test",
            description="Synthetic one-agent scene for obstacle evaluation.",
            n_agents=1,
            obstacles=[Obstacle(center=np.array([100.0, 0.0]), radius=0.1)],
            initial_positions=np.array([[0.0, 0.0]]),
            target_fn=lambda _t: Target(position=np.array([0.0, 0.0]), velocity=np.zeros(2)),
            params=params,
            success_mode="goal_reach",
        )

    def test_evaluate_trial_uses_override_obstacles_for_collisions(self) -> None:
        scenario = self._scenario()
        params = scenario.params
        positions_hist = np.array([[[0.0, 0.0]], [[0.0, 0.0]]])
        target_hist = np.array([[0.0, 0.0], [0.0, 0.0]])
        predicted_target_hist = target_hist.copy()
        u_hist = np.zeros((2, 1, 2))

        original_metrics = evaluate_trial(
            scenario,
            params,
            positions_hist,
            target_hist,
            predicted_target_hist,
            u_hist,
            params.dt,
        )
        jittered_metrics = evaluate_trial(
            scenario,
            params,
            positions_hist,
            target_hist,
            predicted_target_hist,
            u_hist,
            params.dt,
            obstacles=[Obstacle(center=np.array([0.2, 0.0]), radius=0.1)],
        )

        self.assertFalse(original_metrics.collisions)
        self.assertTrue(original_metrics.no_collision)
        self.assertFalse(original_metrics.obs_collision)
        self.assertTrue(jittered_metrics.collisions)
        self.assertFalse(jittered_metrics.no_collision)
        self.assertTrue(jittered_metrics.obs_collision)
        self.assertGreater(original_metrics.min_obs_distance, params.d_obs_safe)
        self.assertLess(jittered_metrics.min_obs_distance, params.d_obs_safe)

    def test_failure_decomposition_fields_are_populated(self) -> None:
        scenario = self._scenario()
        params = scenario.params
        positions_hist = np.array([[[0.0, 0.0]], [[0.0, 0.0]]])
        target_hist = np.array([[0.0, 0.0], [0.0, 0.0]])
        predicted_target_hist = target_hist.copy()
        u_hist = np.zeros((2, 1, 2))

        metrics = evaluate_trial(
            scenario,
            params,
            positions_hist,
            target_hist,
            predicted_target_hist,
            u_hist,
            params.dt,
        )

        self.assertTrue(metrics.gmax_success)
        self.assertTrue(metrics.sigma_success)
        self.assertTrue(metrics.no_collision)
        self.assertTrue(metrics.dwell_success)
        self.assertTrue(metrics.dwell_success_no_collision)
        self.assertIsInstance(metrics.inside_any, bool)
        self.assertIsInstance(metrics.inside_final, bool)
        self.assertGreaterEqual(metrics.time_to_gmax, 0.0)
        self.assertGreaterEqual(metrics.success_hold_time, 0.0)


if __name__ == "__main__":
    unittest.main()
