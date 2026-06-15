from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import Method, Params, SMGFController, Scenario, default_methods
from .metrics import TrialMetrics, evaluate_trial
from .scenarios import build_scenarios


METHOD_LIBRARY = default_methods()
SCENARIOS = build_scenarios()

EXPERIMENT_GROUPS = {
    "A_basic_modules": {
        "title": "Group A Basic Module Verification",
        "entries": [
            {"scene": "a1_single_goal_reach", "methods": ["M1", "M2"]},
            {"scene": "s1_single_obstacle", "methods": ["M1", "M2", "M3"]},
            {"scene": "a3_multi_agent_safety_crossing", "methods": ["M7", "M10"]},
        ],
    },
    "B_encirclement_geometry": {
        "title": "Group B Encirclement Geometry Verification",
        "entries": [
            {"scene": "b1_static_uniform_encirclement", "methods": ["M4", "M7"]},
            {"scene": "s2_same_side_expansion", "methods": ["M4", "M5", "M6", "M7", "M9"]},
            {"scene": "b3_moving_target_open_encirclement", "methods": ["M4", "M7", "M9"]},
        ],
    },
    "C_pressure_passage": {
        "title": "Group C Pressure and Passage Verification",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M4", "M5", "M6", "M7"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M4", "M5", "M6", "M7"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M4", "M5", "M6", "M7"]},
        ],
    },
    "D_prediction_navigation": {
        "title": "Group D Predictive Navigation Verification",
        "entries": [
            {"scene": "s6_fast_target", "methods": ["M4", "M7", "M9"]},
            {"scene": "d2_fast_target_single_obstacle", "methods": ["M4", "M7", "M9"]},
        ],
    },
    "E_integrated_challenges": {
        "title": "Group E Integrated Challenge Verification",
        "entries": [
            {"scene": "e1_tracking_single_obstacle", "methods": ["M4", "M6", "M7", "M9"]},
            {"scene": "e2_tracking_sparse_obstacles", "methods": ["M4", "M6", "M7", "M9"]},
            {"scene": "s5_dense_tracking_easy", "methods": ["M4", "M6", "M7", "M9"]},
            {"scene": "s5_dense_tracking_medium", "methods": ["M4", "M6", "M7", "M9"]},
            {"scene": "s5_dense_tracking_hard", "methods": ["M4", "M6", "M7", "M9"]},
        ],
    },
    "F_safe_tradeoff": {
        "title": "Group F Safety-Force Tradeoff Verification",
        "entries": [
            {"scene": "a3_multi_agent_safety_crossing", "methods": ["M7", "M10"]},
            {"scene": "b3_moving_target_open_encirclement", "methods": ["M7", "M10"]},
            {"scene": "s4_narrow_passage_easy", "methods": ["M7", "M10"]},
            {"scene": "s6_fast_target", "methods": ["M7", "M10"]},
            {"scene": "d2_fast_target_single_obstacle", "methods": ["M7", "M10"]},
        ],
    },
    "G_external_baselines": {
        "title": "Group G External-Style Baseline Comparison",
        "entries": [
            {"scene": "b3_moving_target_open_encirclement", "methods": ["M3", "M11", "M12", "M7", "M9"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M3", "M11", "M12", "M4", "M7"]},
            {"scene": "s6_fast_target", "methods": ["M3", "M11", "M12", "M7"]},
            {"scene": "d2_fast_target_single_obstacle", "methods": ["M3", "M11", "M12", "M4", "M7"]},
            {"scene": "e1_tracking_single_obstacle", "methods": ["M3", "M11", "M12", "M7"]},
        ],
    },
    "H_geo_directional_lite": {
        "title": "Group H Geo-SMGF-lite Directional Prototype",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M13"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M13"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M13"]},
        ],
    },
    "I_branch_basic_validation": {
        "title": "Group I Basic Validation of Three Directions",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M13", "M14", "M15"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M13", "M14", "M15"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M13", "M14", "M15"]},
        ],
    },
    "J_hybrid_basic_validation": {
        "title": "Group J Hybrid Basic Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M13", "M14", "M16"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M13", "M14", "M16"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M13", "M14", "M16"]},
        ],
    },
    "K_scheme_comparison_validation": {
        "title": "Group K Scheme Comparison Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M14", "M16", "M17", "M18"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M14", "M16", "M17", "M18"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M14", "M16", "M17", "M18"]},
        ],
    },
    "L_fusion_basic_validation": {
        "title": "Group L Fusion Basic Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M14", "M16", "M19"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M14", "M16", "M19"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M14", "M16", "M19"]},
        ],
    },
    "M_tensor_basic_validation": {
        "title": "Group M Tensor Basic Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M16", "M18", "M20"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M16", "M18", "M20"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M16", "M18", "M20"]},
        ],
    },
    "N_tensor_energy_polar_validation": {
        "title": "Group N Tensor-Energy-Polar Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M16", "M20", "M21"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M16", "M20", "M21"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M16", "M20", "M21"]},
        ],
    },
    "O_exploration_saved_comparison": {
        "title": "Group O Saved Exploration Comparison",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M13", "M14", "M15", "M16", "M17", "M18", "M19", "M20", "M21"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M13", "M14", "M15", "M16", "M17", "M18", "M19", "M20", "M21"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M13", "M14", "M15", "M16", "M17", "M18", "M19", "M20", "M21"]},
        ],
    },
    "P_continuous_polar_mode_validation": {
        "title": "Group P Continuous Polar Mode Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M16", "M21", "M22"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M16", "M21", "M22"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M16", "M21", "M22"]},
        ],
    },
    "Q_xi_self_localization_validation": {
        "title": "Group Q Xi Self-Localization Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M16", "M22", "M23"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M16", "M22", "M23"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M16", "M22", "M23"]},
        ],
    },
    "R_xi_on_m21_validation": {
        "title": "Group R Xi on M21 Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M16", "M21", "M24"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M16", "M21", "M24"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M16", "M21", "M24"]},
        ],
    },
    "S_orca_lite_comparison": {
        "title": "Group S ORCA Lite Comparison",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25"]},
        ],
    },
    "T_cbf_qp_lite_comparison": {
        "title": "Group T CBF-QP Lite Comparison",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25", "M26"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25", "M26"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25", "M26"]},
        ],
    },
    "U_mpc_cbf_lite_comparison": {
        "title": "Group U MPC-CBF Lite Comparison",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25", "M26", "M27"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25", "M26", "M27"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M4", "M7", "M16", "M23", "M24", "M25", "M26", "M27"]},
        ],
    },
    "V_magnetic_corridor_validation": {
        "title": "Group V Magnetic Corridor Adapter Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M28", "M30", "M31"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M28", "M30", "M31"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M28", "M30", "M31"]},
        ],
    },
    "W_xi_magnetic_polarity_validation": {
        "title": "Group W Xi Magnetic Polarity Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M30", "M31", "M32"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M30", "M31", "M32"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M30", "M31", "M32"]},
        ],
    },
    "X_pure_magnetic_corridor_validation": {
        "title": "Group X Pure Magnetic Corridor Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M30", "M32", "M33"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M30", "M32", "M33"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M30", "M32", "M33"]},
        ],
    },
    "Y_temporal_magnetic_corridor_validation": {
        "title": "Group Y Temporal Magnetic Corridor Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M30", "M33", "M34"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M30", "M33", "M34"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M30", "M33", "M34"]},
        ],
    },
    "Z_magnetic_energy_validation": {
        "title": "Group Z Magnetic Energy Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M30", "M33", "M34", "M35"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M30", "M33", "M34", "M35"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M30", "M33", "M34", "M35"]},
        ],
    },
    "AA_pure_soft_boundary_validation": {
        "title": "Group AA Pure Soft Boundary Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M33", "M34", "M35", "M36"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M33", "M34", "M35", "M36"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M33", "M34", "M35", "M36"]},
        ],
    },
    "AB_queue_occupancy_soft_validation": {
        "title": "Group AB Queue Occupancy Soft Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M35", "M36", "M37"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M35", "M36", "M37"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M35", "M36", "M37"]},
        ],
    },
    "AC_rear_push_soft_validation": {
        "title": "Group AC Rear Push Soft Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M36", "M37", "M38"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M36", "M37", "M38"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M36", "M37", "M38"]},
        ],
    },
    "AD_xi_polarity_axis_validation": {
        "title": "Group AD Xi Polarity Axis Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M35", "M37", "M39"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M35", "M37", "M39"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M35", "M37", "M39"]},
        ],
    },
    "AE_xi_polarity_axis_high_energy_validation": {
        "title": "Group AE Xi Polarity Axis High Energy Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M39", "M40"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M39", "M40"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M39", "M40"]},
        ],
    },
    "AF_xi_polarity_axis_structured_validation": {
        "title": "Group AF Xi Polarity Axis Structured Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M39", "M40", "M41", "M42"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M39", "M40", "M41", "M42"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M39", "M40", "M41", "M42"]},
        ],
    },
    "AG_xi_polarity_axis_high_energy_followups": {
        "title": "Group AG Xi Polarity Axis High Energy Followups",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M41", "M42", "M43"]},
            {"scene": "d_geo_fast_target_lite", "methods": ["M41", "M42", "M43"]},
            {"scene": "e_geo_tracking_single_obstacle_lite", "methods": ["M41", "M42", "M43"]},
        ],
    },
    "AH_hard_energy_adaptation_validation": {
        "title": "Group AH Hard Energy Adaptation Validation",
        "entries": [
            {"scene": "s4_narrow_passage_medium", "methods": ["M41", "M42", "M43", "M46"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M41", "M42", "M43", "M46"]},
        ],
    },
    "M44_m43_harder_validation": {
        "title": "Group M44 M43 Harder Corridor Validation",
        "entries": [
            {"scene": "c_geo_directional_passage_lite", "methods": ["M30", "M43"]},
            {"scene": "s4_narrow_passage_easy", "methods": ["M30", "M43"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M30", "M43"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M30", "M43"]},
        ],
    },
    "AJ_structured_energy_tank_validation": {
        "title": "Group AJ Structured Energy Tank Validation",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M43", "M46", "M47"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M43", "M46", "M47"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M43", "M46", "M47"]},
        ],
    },
    "AK_dual_stage_energy_validation": {
        "title": "Group AK Dual Stage Energy Validation",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M43", "M47", "M48"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M43", "M47", "M48"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M43", "M47", "M48"]},
        ],
    },
    "AL_dual_stage_activation_validation": {
        "title": "Group AL Dual Stage Activation Validation",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M48", "M49", "M50", "M51"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M48", "M49", "M50", "M51"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M48", "M49", "M50", "M51"]},
        ],
    },
    "AM_factorized_energy_ablation_validation": {
        "title": "Group AM Factorized Energy Ablation Validation",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M43", "M52", "M53", "M54", "M55", "M56", "M49"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M43", "M52", "M53", "M54", "M55", "M56", "M49"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M43", "M52", "M53", "M54", "M55", "M56", "M49"]},
        ],
    },
    "AN_entry_energy_propagation_validation": {
        "title": "Group AN Entry Energy Propagation Validation",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M43", "M49", "M57"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M43", "M49", "M57"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M43", "M49", "M57"]},
        ],
    },
    "AO_leader_driven_entry_validation": {
        "title": "Group AO Leader Driven Entry Validation",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M43", "M49", "M57", "M58"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M43", "M49", "M57", "M58"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M43", "M49", "M57", "M58"]},
        ],
    },
    "AP_global_quota_redistribution_validation": {
        "title": "Group AP Global Quota Xi Redistribution Validation",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M43", "M49", "M58", "M59"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M43", "M49", "M58", "M59"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M43", "M49", "M58", "M59"]},
        ],
    },
    "AQ_unified_state_entry_push_validation": {
        "title": "Group AQ Unified State Entry Push Validation",
        "entries": [
            {"scene": "s4_narrow_passage_easy", "methods": ["M43", "M49", "M60"]},
            {"scene": "s4_narrow_passage_medium", "methods": ["M43", "M49", "M60"]},
            {"scene": "s4_narrow_passage_hard", "methods": ["M43", "M49", "M60"]},
        ],
    },
}


def _apply_seed_jitter(scene: Scenario, seed: int) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    positions = scene.initial_positions + rng.normal(scale=0.08, size=scene.initial_positions.shape)
    obstacles = []
    for obs in scene.obstacles:
        jitter = rng.normal(scale=0.03, size=2)
        obstacles.append(type(obs)(center=obs.center + jitter, radius=obs.radius))
    return positions, obstacles


def run_scene_trial(scene: Scenario, method: Method, seed: int = 0, params_override: Params | None = None) -> dict:
    params = params_override or scene.params
    controller = SMGFController(params, method, scene.n_agents)
    positions, obstacles = _apply_seed_jitter(scene, seed)
    steps = int(params.horizon / params.dt) + 1
    omega = np.zeros(scene.n_agents)

    positions_hist = np.zeros((steps, scene.n_agents, 2))
    u_hist = np.zeros((steps, scene.n_agents, 2))
    omega_hist = np.zeros((steps, scene.n_agents))
    rho_hist = np.zeros((steps, scene.n_agents))
    topo_gain_hist = np.zeros((steps, scene.n_agents))
    psi_hist = np.zeros((steps, scene.n_agents))
    phi_hist = np.zeros((steps, scene.n_agents))
    target_hist = np.zeros((steps, 2))
    predicted_target_hist = np.zeros((steps, 2))
    xi_axis_hist = np.zeros((steps, scene.n_agents, 2))
    decision_z_hist = np.zeros((steps, scene.n_agents, 4))
    coverage_aux_hist = np.zeros((steps, scene.n_agents))
    nav_scale_hist = np.ones((steps, scene.n_agents))
    safe_scale_hist = np.ones((steps, scene.n_agents))
    directional_scale_hist = np.ones((steps, scene.n_agents))
    guard_scale_hist = np.ones((steps, scene.n_agents))
    enc_scale_hist = np.ones((steps, scene.n_agents))
    angle_scale_hist = np.ones((steps, scene.n_agents))
    xi_state_hist = np.full((steps, scene.n_agents, 5), np.nan)
    env_axis_hist = np.full((steps, scene.n_agents, 2), np.nan)
    form_axis_hist = np.full((steps, scene.n_agents, 2), np.nan)
    tensor_axis_hist = np.full((steps, scene.n_agents, 2), np.nan)
    env_pressure_hist = np.full((steps, scene.n_agents), np.nan)
    env_anisotropy_hist = np.full((steps, scene.n_agents), np.nan)
    form_anisotropy_hist = np.full((steps, scene.n_agents), np.nan)
    mode_logits_hist = np.full((steps, scene.n_agents, 4), np.nan)
    mode_weights_hist = np.full((steps, scene.n_agents, 4), np.nan)
    directional_axis_hist = np.full((steps, 2), np.nan)
    corridor_activation_hist = np.zeros(steps)
    queue_activation_hist = np.zeros(steps)
    queue_mode_hist = np.zeros(steps)
    structure_lateral_width_hist = np.zeros(steps)
    structure_longitudinal_span_hist = np.zeros(steps)
    global_mode_logits_hist = np.full((steps, 4), np.nan)
    global_mode_weights_hist = np.full((steps, 4), np.nan)
    corridor_confidence_hist = np.full(steps, np.nan)
    corridor_commitment_hist = np.full(steps, np.nan)
    release_need_hist = np.full(steps, np.nan)
    xi_feedback_crowding_mean_hist = np.full(steps, np.nan)
    xi_feedback_crowding_peak_hist = np.full(steps, np.nan)
    xi_feedback_slot_order_readiness_hist = np.full(steps, np.nan)
    xi_feedback_lateral_order_readiness_hist = np.full(steps, np.nan)
    xi_feedback_forward_room_mean_hist = np.full(steps, np.nan)
    xi_feedback_obstacle_block_mean_hist = np.full(steps, np.nan)
    xi_feedback_release_ready_hist = np.full(steps, np.nan)
    guard_release_signal_hist = np.full(steps, np.nan)
    queue_guard_decay_hist = np.full(steps, np.nan)
    directional_release_signal_hist = np.full(steps, np.nan)
    passage_push_strength_hist = np.full(steps, np.nan)
    entry_progress_hist = np.full(steps, np.nan)
    traverse_progress_hist = np.full(steps, np.nan)
    exit_progress_hist = np.full(steps, np.nan)
    forward_free_length_hist = np.full(steps, np.nan)
    obstacle_pressure_hist = np.full(steps, np.nan)
    teammate_pressure_hist = np.full(steps, np.nan)
    post_bottleneck_opening_hist = np.full(steps, np.nan)
    front_blocker_bias_hist = np.full(steps, np.nan)
    structured_energy_tank_hist = np.full(steps, np.nan)
    structured_energy_scene_signal_hist = np.full(steps, np.nan)
    structured_energy_topology_signal_hist = np.full(steps, np.nan)
    structured_energy_store_hist = np.full(steps, np.nan)
    structured_energy_release_hist = np.full(steps, np.nan)
    structured_energy_loss_hist = np.full(steps, np.nan)
    agent_energy_budget_hist = np.full((steps, scene.n_agents), np.nan)
    structured_energy_approach_tank_hist = np.full(steps, np.nan)
    structured_energy_transport_tank_hist = np.full(steps, np.nan)
    structured_energy_convert_hist = np.full(steps, np.nan)
    structured_energy_initial_ready_hist = np.full(steps, np.nan)
    structured_energy_initial_debt_hist = np.full(steps, np.nan)
    spring_chain_storage_hist = np.full(steps, np.nan)
    spring_head_release_hist = np.full(steps, np.nan)
    spring_rear_support_hist = np.full(steps, np.nan)
    spring_link_compression_hist = np.full(steps, np.nan)
    magnetic_drive_intent_hist = np.full((steps, scene.n_agents), np.nan)
    magnetic_release_intent_hist = np.full((steps, scene.n_agents), np.nan)
    magnetic_queue_scale_hist = np.full((steps, scene.n_agents), np.nan)
    magnetic_effective_drive_hist = np.full((steps, scene.n_agents), np.nan)
    magnetic_transport_release_proxy_hist = np.full((steps, scene.n_agents), np.nan)
    unified_r_hist = np.full(steps, np.nan)
    unified_w_hist = np.full(steps, np.nan)
    unified_q_hist = np.full(steps, np.nan)
    unified_a_hist = np.full(steps, np.nan)
    unified_e_hist = np.full(steps, np.nan)
    unified_entry_push_hist = np.full(steps, np.nan)

    for k in range(steps):
        t = k * params.dt
        target = scene.target_fn(t)
        computed = controller.compute(positions, target, omega, obstacles, t, scenario=scene)
        positions_hist[k] = positions
        u_hist[k] = computed["u"]
        omega_hist[k] = computed["omega"]
        rho_hist[k] = computed["rho"]
        topo_gain_hist[k] = computed["topo_gain"]
        psi_hist[k] = computed["psi_tilde"]
        phi_hist[k] = computed["phi"]
        target_hist[k] = target.position
        predicted_target_hist[k] = computed["predicted_target"]
        xi_axis_hist[k] = computed["xi_axis"]
        decision_z_hist[k] = computed["decision_z"]
        coverage_aux_hist[k] = computed["coverage_aux"]
        nav_scale_hist[k] = computed["nav_scale"]
        safe_scale_hist[k] = computed["safe_scale"]
        directional_scale_hist[k] = computed["directional_scale"]
        guard_scale_hist[k] = computed["guard_scale"]
        enc_scale_hist[k] = computed["enc_scale"]
        angle_scale_hist[k] = computed["angle_scale"]
        xi_state_hist[k] = computed["xi_state"]
        env_axis_hist[k] = computed["env_axis"]
        form_axis_hist[k] = computed["form_axis"]
        tensor_axis_hist[k] = computed["tensor_axis"]
        env_pressure_hist[k] = computed["env_pressure"]
        env_anisotropy_hist[k] = computed["env_anisotropy"]
        form_anisotropy_hist[k] = computed["form_anisotropy"]
        mode_logits_hist[k] = computed["mode_logits"]
        mode_weights_hist[k] = computed["mode_weights"]
        directional_axis_hist[k] = computed["directional_axis"]
        corridor_activation_hist[k] = computed["corridor_activation"]
        queue_activation_hist[k] = computed["queue_activation"]
        queue_mode_hist[k] = computed["queue_mode"]
        structure_lateral_width_hist[k] = computed["structure_lateral_width"]
        structure_longitudinal_span_hist[k] = computed["structure_longitudinal_span"]
        global_mode_logits_hist[k] = computed["global_mode_logits"]
        global_mode_weights_hist[k] = computed["global_mode_weights"]
        corridor_confidence_hist[k] = computed["corridor_confidence"]
        corridor_commitment_hist[k] = computed["corridor_commitment"]
        release_need_hist[k] = computed["release_need"]
        xi_feedback_crowding_mean_hist[k] = computed["xi_feedback_crowding_mean"]
        xi_feedback_crowding_peak_hist[k] = computed["xi_feedback_crowding_peak"]
        xi_feedback_slot_order_readiness_hist[k] = computed["xi_feedback_slot_order_readiness"]
        xi_feedback_lateral_order_readiness_hist[k] = computed["xi_feedback_lateral_order_readiness"]
        xi_feedback_forward_room_mean_hist[k] = computed["xi_feedback_forward_room_mean"]
        xi_feedback_obstacle_block_mean_hist[k] = computed["xi_feedback_obstacle_block_mean"]
        xi_feedback_release_ready_hist[k] = computed["xi_feedback_release_ready"]
        guard_release_signal_hist[k] = computed["guard_release_signal"]
        queue_guard_decay_hist[k] = computed["queue_guard_decay"]
        directional_release_signal_hist[k] = computed["directional_release_signal"]
        passage_push_strength_hist[k] = computed["passage_push_strength"]
        entry_progress_hist[k] = computed["entry_progress"]
        traverse_progress_hist[k] = computed["traverse_progress"]
        exit_progress_hist[k] = computed["exit_progress"]
        forward_free_length_hist[k] = computed["forward_free_length"]
        obstacle_pressure_hist[k] = computed.get("obstacle_pressure", np.nan)
        teammate_pressure_hist[k] = computed.get("teammate_pressure", np.nan)
        post_bottleneck_opening_hist[k] = computed["post_bottleneck_opening"]
        front_blocker_bias_hist[k] = computed["front_blocker_bias"]
        structured_energy_tank_hist[k] = computed.get("structured_energy_tank", np.nan)
        structured_energy_scene_signal_hist[k] = computed.get("structured_energy_scene_signal", np.nan)
        structured_energy_topology_signal_hist[k] = computed.get("structured_energy_topology_signal", np.nan)
        structured_energy_store_hist[k] = computed.get("structured_energy_store", np.nan)
        structured_energy_release_hist[k] = computed.get("structured_energy_release", np.nan)
        structured_energy_loss_hist[k] = computed.get("structured_energy_loss", np.nan)
        agent_energy_budget_hist[k] = computed.get("agent_energy_budget", np.full(scene.n_agents, np.nan))
        structured_energy_approach_tank_hist[k] = computed.get("structured_energy_approach_tank", np.nan)
        structured_energy_transport_tank_hist[k] = computed.get("structured_energy_transport_tank", np.nan)
        structured_energy_convert_hist[k] = computed.get("structured_energy_convert", np.nan)
        structured_energy_initial_ready_hist[k] = computed.get("structured_energy_initial_ready", np.nan)
        structured_energy_initial_debt_hist[k] = computed.get("structured_energy_initial_debt", np.nan)
        spring_chain_storage_hist[k] = computed.get("spring_chain_storage", np.nan)
        spring_head_release_hist[k] = computed.get("spring_head_release", np.nan)
        spring_rear_support_hist[k] = computed.get("spring_rear_support", np.nan)
        spring_link_compression_hist[k] = computed.get("spring_link_compression", np.nan)
        magnetic_drive_intent_hist[k] = computed.get("magnetic_drive_intent", np.full(scene.n_agents, np.nan))
        magnetic_release_intent_hist[k] = computed.get("magnetic_release_intent", np.full(scene.n_agents, np.nan))
        magnetic_queue_scale_hist[k] = computed.get("magnetic_queue_scale", np.full(scene.n_agents, np.nan))
        magnetic_effective_drive_hist[k] = computed.get("magnetic_effective_drive", np.full(scene.n_agents, np.nan))
        magnetic_transport_release_proxy_hist[k] = computed.get("magnetic_transport_release_proxy", np.full(scene.n_agents, np.nan))
        unified_r_hist[k] = computed.get("unified_r", np.nan)
        unified_w_hist[k] = computed.get("unified_w", np.nan)
        unified_q_hist[k] = computed.get("unified_q", np.nan)
        unified_a_hist[k] = computed.get("unified_a", np.nan)
        unified_e_hist[k] = computed.get("unified_e", np.nan)
        unified_entry_push_hist[k] = computed.get("unified_entry_push", np.nan)
        omega = computed["omega"]
        if k < steps - 1:
            positions = positions + params.dt * (computed["env"] + computed["u"])

    metrics = evaluate_trial(
        scene,
        params,
        positions_hist,
        target_hist,
        predicted_target_hist,
        u_hist,
        params.dt,
        obstacles=obstacles,
    )
    return {
        "scene": scene,
        "method": method,
        "seed": seed,
        "params": params,
        "obstacles": obstacles,
        "positions_hist": positions_hist,
        "u_hist": u_hist,
        "omega_hist": omega_hist,
        "rho_hist": rho_hist,
        "topo_gain_hist": topo_gain_hist,
        "psi_hist": psi_hist,
        "phi_hist": phi_hist,
        "target_hist": target_hist,
        "predicted_target_hist": predicted_target_hist,
        "xi_axis_hist": xi_axis_hist,
        "decision_z_hist": decision_z_hist,
        "coverage_aux_hist": coverage_aux_hist,
        "nav_scale_hist": nav_scale_hist,
        "safe_scale_hist": safe_scale_hist,
        "directional_scale_hist": directional_scale_hist,
        "guard_scale_hist": guard_scale_hist,
        "enc_scale_hist": enc_scale_hist,
        "angle_scale_hist": angle_scale_hist,
        "xi_state_hist": xi_state_hist,
        "env_axis_hist": env_axis_hist,
        "form_axis_hist": form_axis_hist,
        "tensor_axis_hist": tensor_axis_hist,
        "env_pressure_hist": env_pressure_hist,
        "env_anisotropy_hist": env_anisotropy_hist,
        "form_anisotropy_hist": form_anisotropy_hist,
        "mode_logits_hist": mode_logits_hist,
        "mode_weights_hist": mode_weights_hist,
        "directional_axis_hist": directional_axis_hist,
        "corridor_activation_hist": corridor_activation_hist,
        "queue_activation_hist": queue_activation_hist,
        "queue_mode_hist": queue_mode_hist,
        "structure_lateral_width_hist": structure_lateral_width_hist,
        "structure_longitudinal_span_hist": structure_longitudinal_span_hist,
        "global_mode_logits_hist": global_mode_logits_hist,
        "global_mode_weights_hist": global_mode_weights_hist,
        "corridor_confidence_hist": corridor_confidence_hist,
        "corridor_commitment_hist": corridor_commitment_hist,
        "release_need_hist": release_need_hist,
        "xi_feedback_crowding_mean_hist": xi_feedback_crowding_mean_hist,
        "xi_feedback_crowding_peak_hist": xi_feedback_crowding_peak_hist,
        "xi_feedback_slot_order_readiness_hist": xi_feedback_slot_order_readiness_hist,
        "xi_feedback_lateral_order_readiness_hist": xi_feedback_lateral_order_readiness_hist,
        "xi_feedback_forward_room_mean_hist": xi_feedback_forward_room_mean_hist,
        "xi_feedback_obstacle_block_mean_hist": xi_feedback_obstacle_block_mean_hist,
        "xi_feedback_release_ready_hist": xi_feedback_release_ready_hist,
        "guard_release_signal_hist": guard_release_signal_hist,
        "queue_guard_decay_hist": queue_guard_decay_hist,
        "directional_release_signal_hist": directional_release_signal_hist,
        "passage_push_strength_hist": passage_push_strength_hist,
        "entry_progress_hist": entry_progress_hist,
        "traverse_progress_hist": traverse_progress_hist,
        "exit_progress_hist": exit_progress_hist,
        "forward_free_length_hist": forward_free_length_hist,
        "obstacle_pressure_hist": obstacle_pressure_hist,
        "teammate_pressure_hist": teammate_pressure_hist,
        "post_bottleneck_opening_hist": post_bottleneck_opening_hist,
        "front_blocker_bias_hist": front_blocker_bias_hist,
        "structured_energy_tank_hist": structured_energy_tank_hist,
        "structured_energy_scene_signal_hist": structured_energy_scene_signal_hist,
        "structured_energy_topology_signal_hist": structured_energy_topology_signal_hist,
        "structured_energy_store_hist": structured_energy_store_hist,
        "structured_energy_release_hist": structured_energy_release_hist,
        "structured_energy_loss_hist": structured_energy_loss_hist,
        "agent_energy_budget_hist": agent_energy_budget_hist,
        "structured_energy_approach_tank_hist": structured_energy_approach_tank_hist,
        "structured_energy_transport_tank_hist": structured_energy_transport_tank_hist,
        "structured_energy_convert_hist": structured_energy_convert_hist,
        "structured_energy_initial_ready_hist": structured_energy_initial_ready_hist,
        "structured_energy_initial_debt_hist": structured_energy_initial_debt_hist,
        "spring_chain_storage_hist": spring_chain_storage_hist,
        "spring_head_release_hist": spring_head_release_hist,
        "spring_rear_support_hist": spring_rear_support_hist,
        "spring_link_compression_hist": spring_link_compression_hist,
        "magnetic_drive_intent_hist": magnetic_drive_intent_hist,
        "magnetic_release_intent_hist": magnetic_release_intent_hist,
        "magnetic_queue_scale_hist": magnetic_queue_scale_hist,
        "magnetic_effective_drive_hist": magnetic_effective_drive_hist,
        "magnetic_transport_release_proxy_hist": magnetic_transport_release_proxy_hist,
        "unified_r_hist": unified_r_hist,
        "unified_w_hist": unified_w_hist,
        "unified_q_hist": unified_q_hist,
        "unified_a_hist": unified_a_hist,
        "unified_e_hist": unified_e_hist,
        "unified_entry_push_hist": unified_entry_push_hist,
        "metrics": metrics,
    }


def _plot_trial(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = result["scene"]
    method = result["method"]
    positions_hist = result["positions_hist"]
    target_hist = result["target_hist"]
    predicted_hist = result["predicted_target_hist"]
    omega_hist = result["omega_hist"]
    rho_hist = result["rho_hist"]
    psi_hist = result["psi_hist"]
    dt = result["params"].dt
    time = np.arange(len(positions_hist)) * dt

    fig, ax = plt.subplots(figsize=(8, 6))
    for obs in result["obstacles"]:
        circle = plt.Circle(obs.center, obs.radius, color="gray", alpha=0.3)
        ax.add_patch(circle)
    for i in range(positions_hist.shape[1]):
        ax.plot(positions_hist[:, i, 0], positions_hist[:, i, 1], linewidth=1.4, label=f"agent {i + 1}")
        ax.scatter(positions_hist[0, i, 0], positions_hist[0, i, 1], s=20, marker="o")
        ax.scatter(positions_hist[-1, i, 0], positions_hist[-1, i, 1], s=30, marker="x")
    ax.plot(target_hist[:, 0], target_hist[:, 1], "k--", linewidth=2, label="target")
    ax.plot(predicted_hist[:, 0], predicted_hist[:, 1], color="tab:red", linestyle=":", linewidth=1.5, label="predicted target")
    ax.set_title(f"{scene.title} | {method.name}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(time, np.mean(psi_hist, axis=1), label="mean tilde_psi")
    axes[0].fill_between(time, np.min(psi_hist, axis=1), np.max(psi_hist, axis=1), alpha=0.2)
    axes[0].set_ylabel("tilde_psi")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time, np.mean(omega_hist, axis=1), label="mean Omega", color="tab:orange")
    axes[1].fill_between(time, np.min(omega_hist, axis=1), np.max(omega_hist, axis=1), color="tab:orange", alpha=0.2)
    axes[1].set_ylabel("Omega")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(time, np.mean(rho_hist, axis=1), label="mean rho", color="tab:green")
    axes[2].fill_between(time, np.min(rho_hist, axis=1), np.max(rho_hist, axis=1), color="tab:green", alpha=0.2)
    axes[2].set_ylabel("rho")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.25)
    fig.suptitle(f"State Modulation | {scene.title} | {method.name}")
    fig.tight_layout()
    fig.savefig(output_dir / "state_curves.png", dpi=180)
    plt.close(fig)


def _metrics_to_row(scene_key: str, method_key: str, seed: int, metrics: TrialMetrics) -> dict:
    row = asdict(metrics)
    row.update({"scene": scene_key, "method": method_key, "seed": seed})
    return row


def _entry_trial_task(task: tuple[str, str, int]) -> dict:
    scene_key, method_key, seed = task
    scene = SCENARIOS[scene_key]
    metrics = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed)["metrics"]
    return _metrics_to_row(scene_key, method_key, seed, metrics)


def _summarize_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    return (
        detail_df.groupby(["scene", "method"])
        .agg(
            success_rate=("success", "mean"),
            completion_time_mean=("completion_time", "mean"),
            completion_time_std=("completion_time", "std"),
            collision_rate=("collisions", "mean"),
            obs_collision_rate=("obs_collision", "mean"),
            agent_collision_rate=("agent_collision", "mean"),
            min_obs_distance_mean=("min_obs_distance", "mean"),
            min_agent_distance_mean=("min_agent_distance", "mean"),
            gmax_mean=("max_angle_gap_deg", "mean"),
            radius_var_mean=("radius_variance", "mean"),
            control_cost_mean=("control_cost", "mean"),
            control_smoothness_mean=("control_smoothness", "mean"),
            input_sat_mean=("input_saturation_ratio", "mean"),
            stall_steps_mean=("stall_steps", "mean"),
            lateral_width_final_mean=("lateral_width_final", "mean"),
            longitudinal_span_final_mean=("longitudinal_span_final", "mean"),
            queue_stability_mean=("queue_stability", "mean"),
            inside_any_rate=("inside_any", "mean"),
            inside_final_rate=("inside_final", "mean"),
            gmax_success_rate=("gmax_success", "mean"),
            radius_success_rate=("radius_success", "mean"),
            sigma_success_rate=("sigma_success", "mean"),
            no_collision_rate=("no_collision", "mean"),
            dwell_success_rate=("dwell_success", "mean"),
            dwell_geom_success_rate=("dwell_geom_success", "mean"),
            dwell_success_no_collision_rate=("dwell_success_no_collision", "mean"),
            radius_error_final_mean=("radius_error_final", "mean"),
            success_geom_final_rate=("success_geom_final", "mean"),
            success_no_collision_rate=("success_no_collision", "mean"),
            gmax_reach_time_mean=("gmax_reach_time", "mean"),
            time_to_inside_mean=("time_to_inside", "mean"),
            time_to_gmax_mean=("time_to_gmax", "mean"),
            time_to_radius_mean=("time_to_radius", "mean"),
            time_to_full_geom_mean=("time_to_full_geom", "mean"),
            success_hold_time_mean=("success_hold_time", "mean"),
            post_success_violation_count_mean=("post_success_violation_count", "mean"),
            last_10s_gmax_mean=("last_10s_gmax_mean", "mean"),
            last_10s_radius_error_mean=("last_10s_radius_error_mean", "mean"),
            last_10s_inside_rate_mean=("last_10s_inside_rate", "mean"),
            last_10s_lateral_width_mean=("last_10s_lateral_width_mean", "mean"),
            last_10s_longitudinal_span_mean=("last_10s_longitudinal_span_mean", "mean"),
        )
        .reset_index()
    )


def _run_entries(entries: list[dict], output_path: Path, trials: int, seed_start: int = 0, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    tasks: list[tuple[str, str, int]] = []
    for entry in entries:
        scene_key = entry["scene"]
        scene = SCENARIOS[scene_key]
        for method_key in entry["methods"]:
            if scene.n_agents == 1 and method_key not in {"M1", "M2", "M3"}:
                continue
            result = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed_start)
            plot_dir = output_path / "examples" / scene_key / method_key
            _plot_trial(result, plot_dir)
            for seed in range(seed_start, seed_start + trials):
                tasks.append((scene_key, method_key, seed))

    resolved_workers = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    if resolved_workers <= 1:
        for task in tasks:
            rows.append(_entry_trial_task(task))
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            for row in executor.map(_entry_trial_task, tasks):
                rows.append(row)

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(output_path / "trial_metrics.csv", index=False)
    summary = _summarize_detail(detail_df)
    summary.to_csv(output_path / "summary_metrics.csv", index=False)
    with (output_path / "summary_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)
    return detail_df, summary


def run_suite(output_dir: str | Path, trials: int = 10, scenes: list[str] | None = None, methods: list[str] | None = None, seed_start: int = 0, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_scenes = scenes or [
        "s1_single_obstacle",
        "s2_same_side_expansion",
        "s3_static_encirclement",
        "s4_narrow_passage_easy",
        "s4_narrow_passage_medium",
        "s4_narrow_passage_hard",
        "s5_tracking_stage0",
        "s5_tracking_stage1",
        "s5_tracking_stage2",
        "s6_fast_target",
    ]
    selected_methods = methods or ["M1", "M3", "M4", "M5", "M6", "M7"]
    entries = [{"scene": scene_key, "methods": selected_methods} for scene_key in selected_scenes]
    detail_df, summary = _run_entries(entries, output_path, trials=trials, seed_start=seed_start, workers=workers)

    for scene_key in selected_scenes:
        scene_df = summary[summary["scene"] == scene_key]
        if scene_df.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].bar(scene_df["method"], scene_df["success_rate"], color="tab:blue")
        axes[0].set_title(f"{scene_key} success")
        axes[0].set_ylim(0, 1.05)
        axes[1].bar(scene_df["method"], scene_df["completion_time_mean"], color="tab:orange")
        axes[1].set_title("completion time")
        axes[2].bar(scene_df["method"], scene_df["gmax_mean"], color="tab:green")
        axes[2].set_title("max angle gap [deg]")
        for ax in axes:
            ax.grid(True, alpha=0.2)
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_path / f"{scene_key}_summary.png", dpi=180)
        plt.close(fig)

    return detail_df, summary


def run_group(group_key: str, output_dir: str | Path, trials: int = 10, seed_start: int = 0, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = EXPERIMENT_GROUPS[group_key]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "group_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump({"group_key": group_key, "title": group["title"], "entries": group["entries"], "trials": trials, "seed_start": seed_start, "workers": workers}, fh, ensure_ascii=False, indent=2)
    detail_df, summary = _run_entries(group["entries"], output_path, trials=trials, seed_start=seed_start, workers=workers)
    return detail_df, summary
