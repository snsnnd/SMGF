from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np


EPS = 1e-9


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def unit(vec: np.ndarray, eps: float = EPS) -> np.ndarray:
    length = norm(vec)
    if length <= eps:
        return np.zeros_like(vec)
    return vec / length


def smooth_bound(vec: np.ndarray, limit: float, eps: float = EPS) -> np.ndarray:
    mag = norm(vec)
    if mag <= eps:
        return np.zeros_like(vec)
    return limit * np.tanh(mag / limit) * vec / (mag + eps)


def rotate90(vec: np.ndarray) -> np.ndarray:
    return np.array([-vec[1], vec[0]], dtype=float)


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class Obstacle:
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class Target:
    position: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class Params:
    dt: float = 0.05
    horizon: float = 80.0
    u_max: float = 1.0
    rc: float = 6.0
    d_agent_safe: float = 0.65
    d_obs_safe: float = 0.45
    d_s: float = 2.5
    w_g: float = 1.0
    k_rep: float = 2.8
    f_rep_max: float = 2.5
    f_rep_total_max: float = 3.5
    k_s: float = 3.0
    f_safe_max: float = 3.0
    k_t: float = 1.4
    r0: float = 1.8
    r_c: float = 2.8
    k_r: float = 1.0
    lambda_curl: float = 0.7
    delta: float = 0.2
    eta_min: float = 0.2
    sigma_omega: float = 3.0
    q_psi: float = 2.0
    topo_rho_floor: float = 0.0
    topo_floor_on_threshold: float = 0.35
    topo_floor_off_threshold: float = 0.2
    topo_floor_release_tau: float = 1.0
    tau_omega: float = 0.5
    t_pred: float = 0.7
    t_pred_max: float = 1.5
    gamma_d: float = 1.0
    n_min: int = 3
    beta_lead: float = 0.35
    epsilon_high: float = 1e-3
    epsilon_low: float = 1e-4
    default_curl_sign: int = 1
    gmax_threshold_deg: float = 140.0
    radius_tolerance: float = 0.6
    sigma_r_threshold: float = 0.9
    k_angle: float = 0.35
    geo_queue_gain: float = 1.15
    geo_compact_gain: float = 0.55
    geo_queue_spacing: float = 1.05
    geo_speed_trigger: float = 0.42
    geo_omega_trigger: float = 0.22
    geo_velocity_blend: float = 0.65
    energy_nav_weight: float = 1.0
    energy_rep_weight: float = 1.2
    energy_safe_weight: float = 1.4
    energy_topo_weight: float = 0.9
    energy_enc_weight: float = 0.8
    energy_angle_weight: float = 0.5
    polar_parallel_gain: float = 0.55
    polar_lateral_gain: float = 0.35
    polar_lateral_band: float = 0.75
    hybrid_guard_gain: float = 1.15
    hybrid_guard_spacing: float = 0.9
    matrix_parallel_gain: float = 1.35
    matrix_perp_gain: float = 0.45
    projection_mid_gain: float = 1.0
    projection_low_gain: float = 1.0
    fusion_temperature: float = 1.0
    tensor_parallel_gain: float = 1.45
    tensor_perp_floor: float = 0.28
    tensor_env_keep: float = 0.65
    tensor_polar_env_weight: float = 1.0
    tensor_polar_form_weight: float = 0.8
    tensor_polar_goal_weight: float = 0.7
    tensor_shape_residual_gain: float = 0.6
    polar_mode_temperature: float = 0.8
    xi_center_gain: float = 0.65
    xi_follow_gain: float = 0.85
    xi_yield_gain: float = 0.75
    xi_nav_gain: float = 0.25
    xi_guard_gain: float = 0.55
    orca_agent_horizon: float = 2.5
    orca_obstacle_horizon: float = 2.0
    cbf_gamma: float = 2.2
    cbf_agent_horizon: float = 2.0
    cbf_obstacle_horizon: float = 2.0
    mpc_horizon_steps: int = 8
    mpc_goal_weight: float = 1.0
    mpc_terminal_weight: float = 2.2
    mpc_agent_weight: float = 12.0
    mpc_obstacle_weight: float = 10.0
    mpc_lane_weight: float = 0.55
    mpc_control_weight: float = 0.22
    mpc_heading_span_deg: float = 75.0
    mpc_heading_samples: int = 7
    mpc_speed_levels: tuple[float, ...] = (0.0, 0.5, 1.0)
    magnetic_slot_gain: float = 1.15
    magnetic_lane_gain: float = 0.85
    magnetic_spacing_gain: float = 1.0
    magnetic_lateral_rep_gain: float = 0.4
    magnetic_obstacle_gain: float = 0.85
    magnetic_tangent_gain: float = 0.55
    magnetic_release_gain: float = 0.7
    magnetic_headway_time: float = 1.15
    magnetic_headway_gain: float = 1.35
    magnetic_priority_gain: float = 0.85
    magnetic_pre_release_gain: float = 0.28
    magnetic_release_on: float = 0.58
    magnetic_release_off: float = 0.34
    magnetic_drive_gain: float = 0.95
    magnetic_drive_release_gain: float = 0.65
    magnetic_drive_nav_blend: float = 0.45
    magnetic_approach_gain: float = 1.35
    magnetic_approach_floor: float = 0.22
    magnetic_soft_leader_gain: float = 0.35
    magnetic_soft_curl_boost: float = 1.55
    magnetic_queue_occupancy_gain: float = 1.8
    magnetic_queue_drive_drop_gain: float = 0.92
    magnetic_queue_front_zone: float = 1.2
    magnetic_queue_push_gain: float = 0.75
    magnetic_queue_push_zone: float = 1.15
    magnetic_polarity_lateral_gain: float = 0.42
    magnetic_polarity_gap_gain: float = 0.24
    magnetic_polarity_obstacle_gain: float = 0.32
    magnetic_high_energy_gain: float = 1.8
    magnetic_high_release_gain: float = 1.2
    magnetic_high_curl_boost: float = 2.2
    magnetic_high_approach_floor: float = 0.38
    magnetic_leader_pull_gain: float = 0.9
    magnetic_leader_trail_gain: float = 0.32
    magnetic_leader_span: float = 2.4
    magnetic_adaptive_energy_floor: float = 0.55
    magnetic_adaptive_energy_ceiling: float = 1.7
    magnetic_adaptive_release_ceiling: float = 1.05
    magnetic_adaptive_curl_ceiling: float = 1.6
    magnetic_tank_initial: float = 0.4
    magnetic_tank_capacity: float = 1.0
    magnetic_tank_charge_gain: float = 1.45
    magnetic_tank_release_gain: float = 0.8
    magnetic_tank_loss_gain: float = 0.55
    magnetic_tank_drive_floor: float = 0.75
    magnetic_tank_drive_ceiling: float = 2.1
    magnetic_tank_curl_floor: float = 0.7
    magnetic_tank_curl_ceiling: float = 2.0
    magnetic_approach_tank_capacity: float = 1.0
    magnetic_approach_tank_charge_gain: float = 0.45
    magnetic_approach_tank_loss_gain: float = 0.18
    magnetic_transport_convert_gain: float = 1.1
    magnetic_transport_charge_gain: float = 0.3
    magnetic_transport_release_gain: float = 0.9
    magnetic_transport_loss_gain: float = 0.5
    magnetic_initial_topology_bias: float = 0.35
    magnetic_activation_threshold: float = 0.22
    magnetic_activation_slope: float = 8.0
    magnetic_activation_window_low: float = 0.18
    magnetic_activation_window_high: float = 0.72
    unified_entry_push_gain: float = 2.4
    unified_entry_energy_threshold: float = 0.16
    unified_entry_width_ref: float = 0.75
    unified_entry_slope: float = 10.0
    env_flow: Callable[[np.ndarray, float], np.ndarray] = field(
        default=lambda p, t: np.zeros(2, dtype=float)
    )


@dataclass(frozen=True)
class Method:
    name: str
    baseline_mode: str | None = None
    use_bounded_nav: bool = True
    use_bounded_repulsion: bool = True
    use_curl: bool = True
    use_safe: bool = True
    use_topology: bool = True
    use_psi: bool = True
    use_omega: bool = True
    use_encirclement: bool = True
    force_rho_one: bool = False
    traditional_apf: bool = False
    allow_prediction: bool = True
    use_angle_spread: bool = False
    use_directional_topology: bool = False
    use_energy_flow: bool = False
    use_polar_kernel: bool = False
    use_projection_priority: bool = False
    use_matrix_modulation: bool = False
    use_fusion_attention: bool = False
    use_tensor_modulation: bool = False
    use_tensor_energy_polar: bool = False
    use_continuous_polar_modes: bool = False
    use_self_localization_state: bool = False
    use_global_phase_context: bool = False
    use_corridor_projection_low_level: bool = False
    use_corridor_qp_low_level: bool = False
    use_magnetic_corridor_adapter: bool = False
    use_xi_magnetic_polarity: bool = False
    use_pure_magnetic_corridor_control: bool = False
    use_temporal_magnetic_release: bool = False
    use_magnetic_drive_energy: bool = False
    use_aggressive_magnetic_soft_drive: bool = False
    use_soft_queue_occupancy_rule: bool = False
    use_soft_queue_push_rule: bool = False
    use_xi_polarity_axis_field: bool = False
    use_high_energy_polarity_drive: bool = False
    use_soft_leader_pull_rule: bool = False
    use_adaptive_xi_energy_drive: bool = False
    use_structured_energy_tank: bool = False
    use_dual_stage_energy_tank: bool = False
    dual_stage_activation_mode: str = "linear"
    use_energy_scene_layer: bool = False
    use_energy_topology_layer: bool = False
    use_energy_xi_allocation: bool = False
    use_energy_curl_efficiency: bool = False
    use_energy_activation_trigger: bool = False
    use_entry_energy_propagation: bool = False
    use_energy_global_quota_redistribution: bool = False
    use_leader_driven_entry_bonus: bool = False
    use_front_loaded_transport_release: bool = False
    use_spring_chain_reduced_model: bool = False
    use_unified_state_entry_push: bool = False


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    n_agents: int
    obstacles: list[Obstacle]
    initial_positions: np.ndarray
    target_fn: Callable[[float], Target]
    params: Params
    success_mode: str
    corridor_exit_x: float | None = None


def default_methods() -> dict[str, Method]:
    return {
        "M1": Method("Traditional APF", traditional_apf=True, use_bounded_nav=False, use_bounded_repulsion=False, use_curl=False, use_safe=False, use_topology=False, use_psi=False, use_omega=False, use_encirclement=False, allow_prediction=False),
        "M2": Method("Bounded APF", use_curl=False, use_safe=False, use_topology=False, use_psi=False, use_omega=False, use_encirclement=False, allow_prediction=False),
        "M3": Method("APF-Curl", use_safe=False, use_topology=False, use_psi=False, use_omega=False, use_encirclement=False, allow_prediction=False),
        "M4": Method("Fixed-Topo Guidance", force_rho_one=True),
        "M5": Method("SMGF without Psi", use_psi=False),
        "M6": Method("SMGF without Omega", use_omega=False),
        "M7": Method("Full SMGF"),
        "M8": Method("Full SMGF without Curl", use_curl=False),
        "M9": Method("Full SMGF with Angle Regularization", use_angle_spread=True),
        "M10": Method("Full SMGF without Safe", use_safe=False),
        "M11": Method("Consensus Formation Baseline", baseline_mode="consensus_ring", use_curl=False, use_psi=False, use_omega=False),
        "M12": Method("Leader-Follower Baseline", baseline_mode="leader_follower", use_curl=False, use_topology=False, use_psi=False, use_omega=False, use_angle_spread=False),
        "M13": Method("Geo-SMGF-lite", use_directional_topology=True),
        "M14": Method("EnergyFlow-SMGF-lite", use_energy_flow=True),
        "M15": Method("Polar-SMGF-lite", use_polar_kernel=True),
        "M16": Method("Hybrid-SMGF-lite", use_energy_flow=True, use_directional_topology=True),
        "M17": Method("Projection-SMGF-lite", use_projection_priority=True, use_directional_topology=True),
        "M18": Method("Matrix-Rho-SMGF-lite", use_matrix_modulation=True),
        "M19": Method("Fusion-SMGF-lite", use_fusion_attention=True, use_energy_flow=True, use_directional_topology=True),
        "M20": Method("Tensor-SMGF-lite", use_tensor_modulation=True),
        "M21": Method("Tensor-Energy-Polar-lite", use_tensor_energy_polar=True, use_energy_flow=True, use_directional_topology=True),
        "M22": Method("Continuous-Polar-Mode-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True),
        "M23": Method("Xi-Continuous-Polar-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True),
        "M24": Method("Xi-Tensor-Energy-Polar-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_self_localization_state=True),
        "M25": Method("ORCA-Style Reactive Baseline", baseline_mode="orca_reactive", use_curl=False, use_safe=False, use_topology=False, use_psi=False, use_omega=False, use_angle_spread=False),
        "M26": Method("CBF-QP-Style Safety Filter Baseline", baseline_mode="cbf_qp_filter", use_curl=False, use_safe=False, use_psi=False, use_omega=False, use_angle_spread=False),
        "M27": Method("MPC-CBF-Style Baseline", baseline_mode="mpc_cbf_style", use_curl=False, use_safe=False, use_psi=False, use_omega=False, use_angle_spread=False),
        "M28": Method("Global-Phase-Xi-Continuous-Polar-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True),
        "M29": Method("Global-Phase-Xi-Corridor-Projection-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_corridor_projection_low_level=True),
        "M30": Method("Global-Phase-Xi-Corridor-QP-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_corridor_qp_low_level=True),
        "M31": Method("Magnetic-Corridor-QP-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_corridor_qp_low_level=True, use_magnetic_corridor_adapter=True),
        "M32": Method("Xi-Magnetic-Polarity-QP-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_corridor_qp_low_level=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True),
        "M33": Method("Xi-Magnetic-Only-Corridor-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True),
        "M34": Method("Xi-Magnetic-Temporal-Corridor-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_temporal_magnetic_release=True),
        "M35": Method("Xi-Magnetic-Energy-Corridor-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True),
        "M36": Method("Xi-Magnetic-Aggressive-Energy-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_aggressive_magnetic_soft_drive=True),
        "M37": Method("Xi-Magnetic-Queue-Occupancy-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_aggressive_magnetic_soft_drive=True, use_soft_queue_occupancy_rule=True),
        "M38": Method("Xi-Magnetic-RearPush-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_aggressive_magnetic_soft_drive=True, use_soft_queue_push_rule=True),
        "M39": Method("Xi-Polarity-Axis-Corridor-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True),
        "M40": Method("Xi-Polarity-Axis-HighEnergy-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_high_energy_polarity_drive=True),
        "M41": Method("Xi-Polarity-Axis-Queue-Occupancy-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True),
        "M42": Method("Xi-Polarity-Axis-Leader-Pull-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_leader_pull_rule=True),
        "M43": Method("Xi-Polarity-Axis-Queue-Occupancy-HighEnergy-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_high_energy_polarity_drive=True),
        "M46": Method("Xi-Polarity-Axis-Queue-Occupancy-AdaptiveEnergy-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_adaptive_xi_energy_drive=True),
        "M47": Method("Xi-Polarity-Axis-Queue-Occupancy-TankEnergy-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_structured_energy_tank=True),
        "M48": Method("Xi-Polarity-Axis-Queue-Occupancy-DualTank-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True),
        "M49": Method("Xi-Polarity-Axis-Queue-Occupancy-DualTank-Sigmoid-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="sigmoid", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True),
        "M50": Method("Xi-Polarity-Axis-Queue-Occupancy-DualTank-Tanh-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="tanh", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True),
        "M51": Method("Xi-Polarity-Axis-Queue-Occupancy-DualTank-Window-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="window", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True),
        "M52": Method("Scene-Layer-DualTank-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, use_energy_scene_layer=True),
        "M53": Method("Topology-Layer-DualTank-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, use_energy_topology_layer=True),
        "M54": Method("Xi-Allocation-DualTank-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, use_energy_xi_allocation=True),
        "M55": Method("Curl-Efficiency-DualTank-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, use_energy_curl_efficiency=True),
        "M56": Method("Activation-DualTank-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="sigmoid", use_energy_activation_trigger=True),
        "M57": Method("Leader-Propagated-DualTank-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="sigmoid", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True, use_entry_energy_propagation=True),
        "M58": Method("Leader-Driven-Entry-Propagation-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="sigmoid", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True, use_entry_energy_propagation=True, use_leader_driven_entry_bonus=True),
        "M59": Method("GlobalQuota-Xi-Redistribution-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="sigmoid", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True, use_energy_global_quota_redistribution=True, use_front_loaded_transport_release=True),
        "M61": Method("LeaderFrontLoaded-TransportRelease-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="sigmoid", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True, use_entry_energy_propagation=True, use_leader_driven_entry_bonus=True, use_front_loaded_transport_release=True),
        "M62": Method("SpringChain-Reduced-TransportRelease-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="sigmoid", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True, use_entry_energy_propagation=True, use_leader_driven_entry_bonus=True, use_front_loaded_transport_release=True, use_spring_chain_reduced_model=True),
        "M60": Method("Unified-State-EntryPush-lite", use_tensor_energy_polar=True, use_directional_topology=True, use_energy_flow=True, use_continuous_polar_modes=True, use_self_localization_state=True, use_global_phase_context=True, use_magnetic_corridor_adapter=True, use_xi_magnetic_polarity=True, use_pure_magnetic_corridor_control=True, use_magnetic_drive_energy=True, use_xi_polarity_axis_field=True, use_soft_queue_occupancy_rule=True, use_dual_stage_energy_tank=True, dual_stage_activation_mode="sigmoid", use_energy_scene_layer=True, use_energy_topology_layer=True, use_energy_xi_allocation=True, use_energy_curl_efficiency=True, use_energy_activation_trigger=True, use_unified_state_entry_push=True),
    }


class SMGFController:
    def __init__(self, params: Params, method: Method, n_agents: int):
        self.params = params
        self.method = method
        self.prev_sep = np.tile(np.array([1.0, 0.0]), (n_agents, n_agents, 1))
        self.topo_floor_state = np.zeros(n_agents)
        self.prev_curl_sign = np.array(
            [params.default_curl_sign if i % 2 == 0 else -params.default_curl_sign for i in range(n_agents)],
            dtype=float,
        )
        self.prev_queue_order = np.arange(n_agents)
        self.queue_mode_active = False
        self.prev_baseline_vel = np.zeros((n_agents, 2), dtype=float)
        self.prev_output_vel = np.zeros((n_agents, 2), dtype=float)
        self.magnetic_release_state = np.zeros(n_agents, dtype=float)
        self.magnetic_release_prev = np.zeros(n_agents, dtype=float)
        self.prev_xi_feedback = {
            "crowding_mean": 0.0,
            "crowding_peak": 0.0,
            "slot_order_readiness": 0.0,
            "lateral_order_readiness": 0.0,
            "forward_room_mean": 0.0,
            "obstacle_block_mean": 0.0,
            "release_ready": 0.0,
        }
        self.transport_energy_tank = params.magnetic_tank_initial
        self.approach_energy_tank = 0.0
        self.dual_stage_energy_initialized = False
        self.initial_topology_ready = 0.0
        self.initial_topology_debt = 0.0
        self.last_energy_scene_signal = 0.0
        self.last_energy_topology_signal = 0.0
        self.last_energy_store = 0.0
        self.last_energy_release = 0.0
        self.last_energy_loss = 0.0
        self.last_agent_energy_budget = np.zeros(n_agents, dtype=float)
        self.spring_chain_storage_state = 0.0
        self.last_spring_chain_storage = 0.0
        self.last_spring_head_release = 0.0
        self.last_spring_rear_support = 0.0
        self.last_spring_link_compression = 0.0

    def _consensus_force(self, idx: int, positions: np.ndarray, neighbors: list[int]) -> np.ndarray:
        total = np.zeros(2)
        for j in neighbors:
            rij = positions[idx] - positions[j]
            dij = norm(rij)
            if dij <= 1e-6:
                direction = self.prev_sep[idx, j]
            else:
                direction = rij / dij
                self.prev_sep[idx, j] = direction
            total += self.params.k_t * ((self.params.r0 - dij) / max(self.params.r0, EPS)) * direction
        return total

    def _leader_follower_force(self, idx: int, positions: np.ndarray, target: Target, predicted_target: np.ndarray) -> np.ndarray:
        # Keep the baseline simple and interpretable: one leader pursues the target,
        # followers track a chain behind the leader along the target-motion direction.
        if idx == 0:
            return self.params.w_g * unit(predicted_target - positions[idx])
        heading = unit(predicted_target - target.position)
        if norm(heading) <= EPS:
            heading = unit(predicted_target - positions[0])
        desired = positions[idx - 1] - 0.9 * self.params.r0 * heading
        return self.params.w_g * unit(desired - positions[idx])

    def _predicted_target(self, target: Target) -> np.ndarray:
        horizon = min(self.params.t_pred, self.params.t_pred_max) if self.method.allow_prediction else 0.0
        return target.position + horizon * target.velocity

    def _rotate(self, vec: np.ndarray, angle_rad: float) -> np.ndarray:
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=float)

    def _line_intersection(self, a1: np.ndarray, b1: float, a2: np.ndarray, b2: float) -> np.ndarray | None:
        mat = np.stack([a1, a2], axis=0)
        det = float(np.linalg.det(mat))
        if abs(det) <= 1e-9:
            return None
        return np.linalg.solve(mat, np.array([b1, b2], dtype=float))

    def _line_circle_intersections(self, a: np.ndarray, b: float, radius: float) -> list[np.ndarray]:
        denom = float(a @ a)
        if denom <= EPS:
            return []
        center = (b / denom) * a
        center_norm_sq = float(center @ center)
        radius_sq = radius * radius
        if center_norm_sq > radius_sq + 1e-9:
            return []
        tangent = np.array([-a[1], a[0]], dtype=float)
        tangent_norm = norm(tangent)
        if tangent_norm <= EPS:
            return []
        tangent = tangent / tangent_norm
        offset_sq = max(radius_sq - center_norm_sq, 0.0)
        offset = np.sqrt(offset_sq)
        if offset <= 1e-9:
            return [center]
        return [center + offset * tangent, center - offset * tangent]

    def _feasible_velocity(self, velocity: np.ndarray, constraints: list[tuple[np.ndarray, float]], radius: float) -> bool:
        if norm(velocity) > radius + 1e-7:
            return False
        for normal, limit in constraints:
            if float(normal @ velocity) - limit > 1e-7:
                return False
        return True

    def _solve_velocity_qp(self, preferred: np.ndarray, constraints: list[tuple[np.ndarray, float]], radius: float) -> np.ndarray:
        candidates: list[np.ndarray] = []

        preferred = preferred.astype(float)
        candidates.append(preferred)
        preferred_norm = norm(preferred)
        if preferred_norm > radius:
            candidates.append(radius * preferred / (preferred_norm + EPS))

        for normal, limit in constraints:
            denom = float(normal @ normal)
            if denom <= EPS:
                continue
            projected = preferred - (float(normal @ preferred) - limit) * normal / denom
            candidates.append(projected)
            candidates.extend(self._line_circle_intersections(normal, limit, radius))

        for idx in range(len(constraints)):
            a_i, b_i = constraints[idx]
            for jdx in range(idx + 1, len(constraints)):
                a_j, b_j = constraints[jdx]
                point = self._line_intersection(a_i, b_i, a_j, b_j)
                if point is not None:
                    candidates.append(point)

        best = None
        best_cost = float("inf")
        for velocity in candidates:
            if not self._feasible_velocity(velocity, constraints, radius):
                continue
            cost = float(np.sum((velocity - preferred) ** 2))
            if cost < best_cost:
                best_cost = cost
                best = velocity

        if best is not None:
            return best

        velocity = preferred.copy()
        speed = norm(velocity)
        if speed > radius:
            velocity = radius * velocity / (speed + EPS)
        for _ in range(5):
            updated = False
            for normal, limit in constraints:
                violation = float(normal @ velocity) - limit
                if violation <= 1e-9:
                    continue
                velocity = velocity - violation * normal / max(float(normal @ normal), EPS)
                speed = norm(velocity)
                if speed > radius:
                    velocity = radius * velocity / (speed + EPS)
                updated = True
            if not updated:
                break
        return velocity

    def _orca_preferred_velocity(
        self,
        idx: int,
        positions: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
        success_mode: str,
    ) -> np.ndarray:
        position = positions[idx]
        if success_mode == "encirclement":
            center = (1.0 - self.params.beta_lead) * target.position + self.params.beta_lead * predicted_target
            radial = unit(position - center)
            if norm(radial) <= EPS:
                radial = np.array([1.0, 0.0], dtype=float)
            desired = center + self.params.r_c * radial
            return self.params.w_g * unit(desired - position)
        return self.params.w_g * unit(predicted_target - position)

    def _project_halfplanes(self, preferred: np.ndarray, constraints: list[tuple[np.ndarray, float]]) -> np.ndarray:
        return self._solve_velocity_qp(preferred, constraints, self.params.u_max)

    def _orca_velocity(
        self,
        idx: int,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        neighbor_sets: list[list[int]],
        obstacles: list[Obstacle],
    ) -> np.ndarray:
        constraints: list[tuple[np.ndarray, float]] = []
        tau_agent = max(self.params.orca_agent_horizon, self.params.dt)
        tau_obs = max(self.params.orca_obstacle_horizon, self.params.dt)
        position = positions[idx]

        for j in neighbor_sets[idx]:
            rel = positions[j] - position
            dist = norm(rel)
            sensing_limit = self.params.rc + self.params.u_max * tau_agent + self.params.d_agent_safe
            if dist > sensing_limit:
                continue
            normal = unit(rel)
            if norm(normal) <= EPS:
                normal = -self.prev_sep[idx, j]
            limit = float(normal @ self.prev_baseline_vel[j]) + (dist - self.params.d_agent_safe) / tau_agent
            constraints.append((normal, limit))

        for obs in obstacles:
            delta = position - obs.center
            center_dist = norm(delta)
            clearance = center_dist - obs.radius
            sensing_limit = self.params.d_s + self.params.u_max * tau_obs
            if clearance > sensing_limit:
                continue
            normal = unit(delta)
            if norm(normal) <= EPS:
                normal = np.array([1.0, 0.0], dtype=float)
            limit = (clearance - self.params.d_obs_safe) / tau_obs
            constraints.append((-normal, limit))

        return self._project_halfplanes(preferred_velocities[idx], constraints)

    def _cbf_nominal_velocity(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        target: Target,
        predicted_target: np.ndarray,
        success_mode: str,
    ) -> np.ndarray:
        rep, _phi = self._bounded_repulsion(positions[idx])
        nav = self.params.w_g * unit(predicted_target - positions[idx])
        feedforward = 0.25 * target.velocity
        if success_mode == "encirclement":
            topo = self._consensus_force(idx, positions, neighbors)
            enc = self._encirclement_force(positions[idx], target, predicted_target)
            return smooth_bound(nav + feedforward + 0.6 * rep + 0.8 * topo + enc, self.params.u_max)
        if success_mode == "corridor_pass":
            topo = self._consensus_force(idx, positions, neighbors)
            return smooth_bound(nav + 0.9 * rep + 0.55 * topo, self.params.u_max)
        if len(positions) == 1:
            return smooth_bound(nav + 0.9 * rep + feedforward, self.params.u_max)
        lead = self._leader_follower_force(idx, positions, target, predicted_target)
        enc = 0.55 * self._encirclement_force(positions[idx], target, predicted_target)
        topo = 0.45 * self._consensus_force(idx, positions, neighbors)
        return smooth_bound(lead + feedforward + 0.6 * rep + topo + enc, self.params.u_max)

    def _cbf_velocity(
        self,
        idx: int,
        positions: np.ndarray,
        nominal_velocities: np.ndarray,
        filtered_velocities: np.ndarray,
        neighbor_sets: list[list[int]],
        obstacles: list[Obstacle],
    ) -> np.ndarray:
        constraints: list[tuple[np.ndarray, float]] = []
        gamma = self.params.cbf_gamma
        tau_agent = max(self.params.cbf_agent_horizon, self.params.dt)
        tau_obs = max(self.params.cbf_obstacle_horizon, self.params.dt)
        position = positions[idx]

        for j in neighbor_sets[idx]:
            rel = positions[j] - position
            dist = norm(rel)
            if dist > self.params.rc + self.params.u_max * tau_agent + self.params.d_agent_safe:
                continue
            normal = unit(rel)
            if norm(normal) <= EPS:
                normal = -self.prev_sep[idx, j]
            h = dist * dist - self.params.d_agent_safe * self.params.d_agent_safe
            other_vel = 0.5 * (filtered_velocities[j] + nominal_velocities[j])
            limit = gamma * h - 2.0 * float(rel @ other_vel)
            constraints.append((-2.0 * rel, limit))

        for obs in obstacles:
            delta = position - obs.center
            center_dist = norm(delta)
            if center_dist <= EPS:
                continue
            effective_radius = obs.radius + self.params.d_obs_safe
            clearance = center_dist - effective_radius
            if clearance > self.params.d_s + self.params.u_max * tau_obs:
                continue
            h = center_dist * center_dist - effective_radius * effective_radius
            constraints.append((-2.0 * delta, gamma * h))

        return self._project_halfplanes(nominal_velocities[idx], constraints)

    def _cbf_filter_velocities(
        self,
        positions: np.ndarray,
        nominal_velocities: np.ndarray,
        neighbor_sets: list[list[int]],
        obstacles: list[Obstacle],
    ) -> np.ndarray:
        filtered = nominal_velocities.copy()
        for _ in range(3):
            updated = filtered.copy()
            for i in range(len(positions)):
                updated[i] = self._cbf_velocity(i, positions, nominal_velocities, filtered, neighbor_sets, obstacles)
            filtered = updated
        return filtered

    def _queue_rank_map(self, positions: np.ndarray, axis: np.ndarray) -> np.ndarray:
        center = np.mean(positions, axis=0)
        projections = (positions - center[None, :]) @ axis
        order = np.argsort(-projections, kind="stable")
        ranks = np.zeros(len(order), dtype=int)
        ranks[order] = np.arange(len(order))
        return ranks

    def _mpc_reference_velocity(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        target: Target,
        predicted_target: np.ndarray,
        success_mode: str,
        axis: np.ndarray,
        ranks: np.ndarray,
    ) -> np.ndarray:
        position = positions[idx]
        rep, _phi = self._bounded_repulsion(position)
        topo = 0.35 * self._consensus_force(idx, positions, neighbors)
        feedforward = 0.3 * target.velocity
        axis = unit(axis)
        if norm(axis) <= EPS:
            axis = np.array([1.0, 0.0], dtype=float)
        perp = rotate90(axis)
        rank = int(ranks[idx])
        if success_mode == "encirclement":
            center = (1.0 - self.params.beta_lead) * target.position + self.params.beta_lead * predicted_target
            radial = unit(position - center)
            if norm(radial) <= EPS:
                radial = unit(self._rotate(axis, 2.0 * np.pi * rank / max(len(positions), 1)))
            desired = center + self.params.r_c * radial
            enc = 0.8 * self._encirclement_force(position, target, predicted_target)
            return smooth_bound(self.params.w_g * unit(desired - position) + 0.55 * rep + topo + enc + feedforward, self.params.u_max)

        slot_offset = (rank + 0.35) * 0.88 * self.params.r0
        slot = predicted_target - slot_offset * axis
        lateral_error = float((position - slot) @ perp)
        lane_term = -0.35 * lateral_error * perp
        nav = self.params.w_g * unit(slot - position)
        if success_mode == "corridor_pass":
            return smooth_bound(nav + 0.85 * rep + 0.55 * topo + lane_term, self.params.u_max)
        enc = 0.45 * self._encirclement_force(position, target, predicted_target)
        return smooth_bound(nav + 0.55 * rep + topo + enc + feedforward + lane_term, self.params.u_max)

    def _mpc_candidate_velocities(self, preferred: np.ndarray, fallback_axis: np.ndarray) -> list[np.ndarray]:
        axis = unit(preferred)
        if norm(axis) <= EPS:
            axis = unit(fallback_axis)
        if norm(axis) <= EPS:
            axis = np.array([1.0, 0.0], dtype=float)
        heading_offsets = np.deg2rad(np.linspace(-self.params.mpc_heading_span_deg, self.params.mpc_heading_span_deg, self.params.mpc_heading_samples))
        candidates: list[np.ndarray] = [np.zeros(2, dtype=float), smooth_bound(preferred, self.params.u_max)]
        for offset in heading_offsets:
            direction = unit(self._rotate(axis, float(offset)))
            if norm(direction) <= EPS:
                continue
            for scale in self.params.mpc_speed_levels:
                candidates.append(float(scale) * self.params.u_max * direction)
        return candidates

    def _mpc_safety_penalty(self, clearance: float, margin: float, weight: float) -> float:
        if clearance >= margin:
            return 0.0
        gap = margin - clearance
        penalty = weight * gap * gap
        if clearance < 0.0:
            penalty += 50.0 * abs(clearance)
        return penalty

    def _mpc_rollout_cost(
        self,
        idx: int,
        candidate_velocity: np.ndarray,
        positions: np.ndarray,
        other_nominals: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
        scenario: Scenario,
        t: float,
        axis: np.ndarray,
        ranks: np.ndarray,
    ) -> float:
        del predicted_target  # future target is recomputed along the horizon
        dt = self.params.dt
        horizon_steps = max(1, self.params.mpc_horizon_steps)
        perp = rotate90(axis)
        rank = int(ranks[idx])
        future_self = positions[idx].copy()
        cost = 0.0
        for step in range(1, horizon_steps + 1):
            future_t = t + step * dt
            target_k = scenario.target_fn(future_t)
            predicted_k = target_k.position + min(self.params.t_pred, self.params.t_pred_max) * target_k.velocity
            axis_k = unit(target_k.velocity)
            if norm(axis_k) <= EPS:
                axis_k = unit(predicted_k - future_self)
            if norm(axis_k) <= EPS:
                axis_k = axis
            perp_k = rotate90(axis_k)
            future_self = future_self + dt * candidate_velocity
            others_k = positions + step * dt * other_nominals

            if scenario.success_mode == "corridor_pass":
                slot = predicted_k - (rank + 0.35) * 0.9 * self.params.r0 * axis_k
                goal_cost = norm(future_self - slot) + self.params.mpc_lane_weight * abs(float((future_self - slot) @ perp_k))
            elif scenario.success_mode == "encirclement":
                radial_error = abs(norm(future_self - predicted_k) - self.params.r_c)
                goal_cost = radial_error + 0.25 * abs(float((future_self - predicted_k) @ perp_k))
            else:
                slot = predicted_k - (rank + 0.35) * 0.85 * self.params.r0 * axis_k
                goal_cost = norm(future_self - slot) + 0.25 * norm(future_self - target_k.position) + self.params.mpc_lane_weight * abs(float((future_self - slot) @ perp_k))

            min_agent_clearance = float("inf")
            for j in range(len(positions)):
                if j == idx:
                    continue
                min_agent_clearance = min(min_agent_clearance, norm(future_self - others_k[j]) - self.params.d_agent_safe)
            if not np.isfinite(min_agent_clearance):
                min_agent_clearance = 10.0

            min_obs_clearance = float("inf")
            for obs in scenario.obstacles:
                min_obs_clearance = min(min_obs_clearance, norm(future_self - obs.center) - (obs.radius + self.params.d_obs_safe))
            if not np.isfinite(min_obs_clearance):
                min_obs_clearance = 10.0

            stage_weight = 1.0 + 0.08 * step
            cost += stage_weight * self.params.mpc_goal_weight * goal_cost
            cost += stage_weight * self._mpc_safety_penalty(min_agent_clearance, 0.18, self.params.mpc_agent_weight)
            cost += stage_weight * self._mpc_safety_penalty(min_obs_clearance, 0.15, self.params.mpc_obstacle_weight)

        cost += self.params.mpc_terminal_weight * goal_cost
        cost += self.params.mpc_control_weight * float(np.sum((candidate_velocity - self.prev_baseline_vel[idx]) ** 2))
        return cost

    def _mpc_nominal_velocities(
        self,
        positions: np.ndarray,
        neighbors: list[list[int]],
        target: Target,
        predicted_target: np.ndarray,
        scenario: Scenario,
        t: float,
    ) -> np.ndarray:
        axis = self._directional_axis(positions, target, predicted_target)
        ranks = self._queue_rank_map(positions, axis)
        base_nominals = np.zeros_like(positions)
        for i in range(len(positions)):
            base_nominals[i] = self._mpc_reference_velocity(i, positions, neighbors[i], target, predicted_target, scenario.success_mode, axis, ranks)

        nominal = base_nominals.copy()
        for _ in range(2):
            updated = nominal.copy()
            for i in range(len(positions)):
                candidates = self._mpc_candidate_velocities(nominal[i], axis)
                best_velocity = nominal[i]
                best_cost = float("inf")
                for candidate in candidates:
                    others = nominal.copy()
                    others[i] = candidate
                    candidate_cost = self._mpc_rollout_cost(i, candidate, positions, others, target, predicted_target, scenario, t, axis, ranks)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_velocity = candidate
                updated[i] = best_velocity
            nominal = updated
        return nominal

    def _neighbor_sets(self, positions: np.ndarray) -> list[list[int]]:
        neighbors: list[list[int]] = []
        for i, pi in enumerate(positions):
            local = []
            for j, pj in enumerate(positions):
                if i == j:
                    continue
                if norm(pj - pi) <= self.params.rc:
                    local.append(j)
            neighbors.append(local)
        return neighbors

    def _obstacle_distance(self, position: np.ndarray, obstacle: Obstacle) -> tuple[float, np.ndarray]:
        delta = position - obstacle.center
        center_dist = norm(delta)
        return center_dist - obstacle.radius, unit(delta)

    def _traditional_repulsion(self, position: np.ndarray, obstacle: Obstacle) -> np.ndarray:
        d, normal = self._obstacle_distance(position, obstacle)
        if d >= self.params.d_s or d <= EPS:
            return np.zeros(2)
        scale = self.params.k_rep * (1.0 / max(d, 1e-3) - 1.0 / self.params.d_s) / max(d * d, 1e-3)
        return scale * normal

    def _bounded_repulsion(self, position: np.ndarray) -> tuple[np.ndarray, float]:
        total = np.zeros(2)
        phi = 0.0
        for obstacle in self.params_obstacles:
            d, normal = self._obstacle_distance(position, obstacle)
            xi = clip01((self.params.d_s - d) / max(self.params.d_s - self.params.d_obs_safe, EPS))
            total += self.params.f_rep_max * (xi ** 2) * normal
            phi += max(0.0, (self.params.d_s - d) / max(self.params.d_s, EPS))
        return smooth_bound(total, self.params.f_rep_total_max), clip01(phi)

    def _curl_term(self, idx: int, position: np.ndarray, rep: np.ndarray, predicted_target: np.ndarray) -> np.ndarray:
        if not self.method.use_curl:
            return np.zeros(2)
        a = float((predicted_target - position) @ rotate90(rep))
        if abs(a) > self.params.epsilon_high:
            self.prev_curl_sign[idx] = np.sign(a)
        elif abs(a) <= self.params.epsilon_low:
            self.prev_curl_sign[idx] = self.params.default_curl_sign if idx % 2 == 0 else -self.params.default_curl_sign
        return self.params.lambda_curl * self.prev_curl_sign[idx] * rotate90(rep)

    def _safe_force(self, idx: int, positions: np.ndarray, neighbors: list[int]) -> np.ndarray:
        if not self.method.use_safe:
            return np.zeros(2)
        total = np.zeros(2)
        for j in neighbors:
            rij = positions[idx] - positions[j]
            dij = norm(rij)
            if dij > 1e-6:
                direction = rij / dij
                self.prev_sep[idx, j] = direction
            else:
                direction = self.prev_sep[idx, j]
            chi = max(0.0, (self.params.d_agent_safe - dij) / max(self.params.d_agent_safe, EPS))
            total += self.params.k_s * chi * direction
        return smooth_bound(total, self.params.f_safe_max)

    def _topology_force(self, idx: int, positions: np.ndarray, neighbors: list[int]) -> np.ndarray:
        if not self.method.use_topology:
            return np.zeros(2)
        total = np.zeros(2)
        for j in neighbors:
            rij = positions[idx] - positions[j]
            dij = norm(rij)
            if dij <= 1e-6:
                direction = self.prev_sep[idx, j]
            else:
                direction = rij / dij
                self.prev_sep[idx, j] = direction
            total += self.params.k_t * np.tanh((self.params.r0 - dij) / max(self.params.r0, EPS)) * direction
        return total

    def _directional_axis(self, positions: np.ndarray, target: Target, predicted_target: np.ndarray) -> np.ndarray:
        center = np.mean(positions, axis=0)
        nav_axis = unit(predicted_target - center)
        vel_axis = unit(target.velocity)
        if norm(vel_axis) <= EPS:
            return nav_axis if norm(nav_axis) > EPS else np.array([1.0, 0.0], dtype=float)
        if norm(nav_axis) <= EPS:
            return vel_axis
        blended = self.params.geo_velocity_blend * vel_axis + (1.0 - self.params.geo_velocity_blend) * nav_axis
        axis = unit(blended)
        return axis if norm(axis) > EPS else vel_axis

    def _directional_activation(self, omega: np.ndarray, target: Target) -> tuple[float, float]:
        omega_peak = float(np.max(omega)) if len(omega) else 0.0
        corridor_activation = clip01((omega_peak - self.params.geo_omega_trigger) / max(1.0 - self.params.geo_omega_trigger, EPS))
        target_speed = norm(target.velocity)
        queue_activation = clip01((target_speed - self.params.geo_speed_trigger) / max(self.params.geo_speed_trigger, EPS))
        return corridor_activation, queue_activation

    def _directional_reconfigure_force(
        self,
        positions: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
        omega: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if not self.method.use_directional_topology:
            return np.zeros_like(positions), 0.0
        corridor_activation, queue_activation = self._directional_activation(omega, target)
        queue_mode = max(corridor_activation, queue_activation)
        if queue_mode <= EPS:
            self.queue_mode_active = False
            return np.zeros_like(positions), 0.0
        axis = self._directional_axis(positions, target, predicted_target)
        axis = unit(axis)
        if norm(axis) <= EPS:
            self.queue_mode_active = False
            return np.zeros_like(positions), 0.0
        perp_axis = rotate90(axis)
        center = np.mean(positions, axis=0)
        parallel_coords = np.array([float((position - center) @ axis) for position in positions])
        lateral_coords = np.array([float((position - center) @ perp_axis) for position in positions])

        if not self.queue_mode_active:
            self.prev_queue_order = np.argsort(-parallel_coords, kind="stable")
            self.queue_mode_active = True
        sorted_indices = self.prev_queue_order.copy()

        centered_slots = self.params.geo_queue_spacing * (0.5 * (len(positions) - 1) - np.arange(len(positions), dtype=float))
        desired_parallel = np.zeros(len(positions))
        desired_parallel[sorted_indices] = centered_slots
        compact_activation = max(corridor_activation, 0.45 * queue_activation)

        reconfigure_force = np.zeros_like(positions)
        for i in range(len(positions)):
            parallel_error = desired_parallel[i] - parallel_coords[i]
            queue_term = queue_mode * self.params.geo_queue_gain * np.tanh(parallel_error / max(self.params.geo_queue_spacing, EPS)) * axis
            compact_term = -compact_activation * self.params.geo_compact_gain * np.tanh(lateral_coords[i] / max(0.45 * self.params.r0, EPS)) * perp_axis
            reconfigure_force[i] = queue_term + compact_term
        return reconfigure_force, queue_mode

    def _blend_force(self, force: np.ndarray, weight: float) -> np.ndarray:
        magnitude = norm(force)
        if magnitude <= EPS or weight <= 0.0:
            return np.zeros(2)
        return weight * np.tanh(magnitude) * force / (magnitude + EPS)

    def _softmax(self, values: np.ndarray, temperature: float | None = None) -> np.ndarray:
        if len(values) == 0:
            return values
        temp = self.params.fusion_temperature if temperature is None else max(temperature, EPS)
        shifted = (values - np.max(values)) / temp
        exp_values = np.exp(np.clip(shifted, -40.0, 40.0))
        return exp_values / max(float(np.sum(exp_values)), EPS)

    def _directional_matrix(self, axis: np.ndarray, parallel_gain: float, perp_gain: float) -> np.ndarray:
        axis = unit(axis)
        if norm(axis) <= EPS:
            return np.eye(2)
        perp_axis = rotate90(axis)
        basis = np.column_stack((axis, perp_axis))
        gains = np.diag([parallel_gain, perp_gain])
        return basis @ gains @ basis.T

    def _nullspace_project(self, force: np.ndarray, blocker: np.ndarray) -> np.ndarray:
        if norm(blocker) <= EPS:
            return force
        direction = unit(blocker)
        projector = np.eye(2) - np.outer(direction, direction)
        return projector @ force

    def _principal_axis(self, matrix: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        fallback = unit(fallback)
        if norm(fallback) <= EPS:
            fallback = np.array([1.0, 0.0], dtype=float)
        evals, evecs = np.linalg.eigh(matrix)
        axis = evecs[:, int(np.argmax(evals))]
        if float(axis @ fallback) < 0.0:
            axis = -axis
        return axis

    def _empty_understanding_details(self) -> dict[str, np.ndarray | float]:
        return {
            "env_axis": np.full(2, np.nan, dtype=float),
            "form_axis": np.full(2, np.nan, dtype=float),
            "tensor_axis": np.full(2, np.nan, dtype=float),
            "env_pressure": np.nan,
            "env_anisotropy": np.nan,
            "form_anisotropy": np.nan,
            "mode_logits": np.full(4, np.nan, dtype=float),
            "mode_weights": np.full(4, np.nan, dtype=float),
        }

    def _safe_nanmean(self, values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return 0.0
        return float(np.nanmean(arr))

    def _safe_nanmax(self, values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return 0.0
        return float(np.nanmax(arr))

    def _aggregate_xi_feedback(self, xi_states: np.ndarray) -> dict[str, float]:
        if xi_states.size == 0 or np.all(np.isnan(xi_states)):
            return self.prev_xi_feedback.copy()
        slot_error = np.abs(xi_states[:, 0])
        lateral_offset = np.abs(xi_states[:, 1])
        gap_balance = xi_states[:, 2]
        obs_bias = xi_states[:, 3]
        crowding = xi_states[:, 4]
        slot_order_readiness = clip01(1.0 - self._safe_nanmean(slot_error))
        lateral_order_readiness = clip01(1.0 - self._safe_nanmean(lateral_offset))
        forward_room_mean = self._safe_nanmean(np.clip(0.5 * (1.0 + gap_balance), 0.0, 1.0))
        obstacle_block_mean = self._safe_nanmean(np.clip(0.5 * (1.0 - obs_bias), 0.0, 1.0))
        crowding_mean = self._safe_nanmean(crowding)
        crowding_peak = self._safe_nanmax(crowding)
        release_ready = clip01(
            0.3 * slot_order_readiness
            + 0.25 * lateral_order_readiness
            + 0.2 * forward_room_mean
            + 0.15 * (1.0 - obstacle_block_mean)
            + 0.1 * crowding_mean
        )
        return {
            "crowding_mean": crowding_mean,
            "crowding_peak": crowding_peak,
            "slot_order_readiness": slot_order_readiness,
            "lateral_order_readiness": lateral_order_readiness,
            "forward_room_mean": forward_room_mean,
            "obstacle_block_mean": obstacle_block_mean,
            "release_ready": release_ready,
        }

    def _global_structure_summary(self, positions: np.ndarray, target: Target, predicted_target: np.ndarray) -> dict[str, np.ndarray | float]:
        axis = self._directional_axis(positions, target, predicted_target)
        axis = unit(axis)
        if norm(axis) <= EPS:
            axis = np.array([1.0, 0.0], dtype=float)
        perp_axis = rotate90(axis)
        center = np.mean(positions, axis=0)
        parallel_coords = np.array([float((position - center) @ axis) for position in positions])
        lateral_coords = np.array([float((position - center) @ perp_axis) for position in positions])
        queue_order = np.argsort(-parallel_coords, kind="stable")
        centered_slots = self.params.geo_queue_spacing * (0.5 * (len(positions) - 1) - np.arange(len(positions), dtype=float))
        desired_parallel = np.zeros(len(positions))
        desired_parallel[queue_order] = centered_slots
        return {
            "axis": axis,
            "perp_axis": perp_axis,
            "center": center,
            "parallel_coords": parallel_coords,
            "lateral_coords": lateral_coords,
            "queue_order": queue_order,
            "desired_parallel": desired_parallel,
            "lateral_width": float(np.max(lateral_coords) - np.min(lateral_coords)) if len(lateral_coords) else 0.0,
            "longitudinal_span": float(np.max(parallel_coords) - np.min(parallel_coords)) if len(parallel_coords) else 0.0,
        }

    def _global_corridor_features(
        self,
        positions: np.ndarray,
        obstacles: list[Obstacle],
        structure: dict[str, np.ndarray | float],
        scene: Scenario | None,
    ) -> dict[str, float]:
        center = np.asarray(structure["center"], dtype=float)
        axis = np.asarray(structure["axis"], dtype=float)
        perp_axis = np.asarray(structure["perp_axis"], dtype=float)
        lateral_width = float(structure["lateral_width"])
        queue_order = np.asarray(structure["queue_order"], dtype=int)
        parallel_coords = np.asarray(structure["parallel_coords"], dtype=float)

        lane_half_width = max(0.5 * lateral_width, 0.8 * self.params.r0)
        default_forward_clearance = 3.5 * self.params.r0
        forward_clearance = default_forward_clearance
        obstacle_pressure = 0.0
        for obstacle in obstacles:
            rel = obstacle.center - center
            forward = float(rel @ axis)
            lateral = abs(float(rel @ perp_axis))
            effective_radius = obstacle.radius + self.params.d_obs_safe
            if lateral > lane_half_width + effective_radius:
                continue
            if forward >= -effective_radius:
                forward_clearance = min(forward_clearance, max(forward - effective_radius, 0.0))
            if forward >= 0.0:
                obstacle_pressure = max(
                    obstacle_pressure,
                    clip01((self.params.d_s - max(forward - effective_radius, 0.0)) / max(self.params.d_s, EPS)),
                )

        teammate_pressure = 0.0
        if len(queue_order) > 1:
            gaps = parallel_coords[queue_order[:-1]] - parallel_coords[queue_order[1:]]
            min_gap = float(np.min(gaps)) if len(gaps) else self.params.hybrid_guard_spacing
            teammate_pressure = clip01((self.params.hybrid_guard_spacing - min_gap) / max(self.params.hybrid_guard_spacing, EPS))

        exit_progress = 0.0
        entry_progress = 0.0
        traverse_progress = 0.0
        release_progress = 0.0
        post_bottleneck_opening = 0.0
        if scene is not None and scene.corridor_exit_x is not None:
            front_x = float(np.max(positions[:, 0]))
            if obstacles:
                corridor_start_x = min(float(obstacle.center[0] - obstacle.radius) for obstacle in obstacles)
            else:
                corridor_start_x = scene.corridor_exit_x - 2.0 * self.params.r0
            corridor_end_x = scene.corridor_exit_x
            entry_progress = clip01((front_x - (corridor_start_x - 1.0 * self.params.r0)) / max(1.8 * self.params.r0, EPS))
            traverse_progress = clip01((front_x - corridor_start_x) / max(corridor_end_x - corridor_start_x, EPS))
            exit_progress = clip01((front_x - (scene.corridor_exit_x - 2.0 * self.params.r0)) / max(2.5 * self.params.r0, EPS))
            release_progress = clip01((front_x - (scene.corridor_exit_x - 1.2 * self.params.r0)) / max(1.8 * self.params.r0, EPS))
            post_bottleneck_opening = clip01((front_x - scene.corridor_exit_x) / max(1.5 * self.params.r0, EPS))

        front_blocker_bias = float(np.clip(obstacle_pressure - teammate_pressure, -1.0, 1.0))
        return {
            "forward_free_length": float(forward_clearance),
            "obstacle_pressure": float(obstacle_pressure),
            "teammate_pressure": float(teammate_pressure),
            "entry_progress": float(entry_progress),
            "traverse_progress": float(traverse_progress),
            "exit_progress": float(exit_progress),
            "release_progress": float(release_progress),
            "post_bottleneck_opening": float(post_bottleneck_opening),
            "front_blocker_bias": front_blocker_bias,
        }

    def _global_phase_context(
        self,
        positions: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
        omega: np.ndarray,
        env_pressures: np.ndarray,
        env_anisotropies: np.ndarray,
        form_anisotropies: np.ndarray,
        scene: Scenario | None,
        xi_feedback: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray | float]:
        feedback = self.prev_xi_feedback if xi_feedback is None else xi_feedback
        structure = self._global_structure_summary(positions, target, predicted_target)
        corridor_activation, queue_activation = self._directional_activation(omega, target)
        queue_mode = max(corridor_activation, queue_activation)
        corridor_features = self._global_corridor_features(positions, self.params_obstacles, structure, scene)
        speed_norm = clip01(norm(target.velocity) / max(self.params.geo_speed_trigger, EPS))
        env_pressure_mean = self._safe_nanmean(env_pressures)
        env_pressure_peak = self._safe_nanmax(env_pressures)
        env_anisotropy_mean = self._safe_nanmean(env_anisotropies)
        form_anisotropy_mean = self._safe_nanmean(form_anisotropies)
        open_space = clip01(1.0 - env_pressure_peak)
        corridor_confidence = clip01(
            0.35 * corridor_activation
            + 0.2 * env_pressure_peak
            + 0.15 * env_anisotropy_mean
            + 0.15 * corridor_features["obstacle_pressure"]
            + 0.12 * clip01(0.5 * (corridor_features["front_blocker_bias"] + 1.0))
            + 0.12 * corridor_features["entry_progress"]
            + 0.08 * corridor_features["traverse_progress"]
            + 0.08 * feedback["crowding_peak"]
            - 0.25 * corridor_features["post_bottleneck_opening"]
        )
        corridor_commitment = clip01(
            0.45 * corridor_confidence
            + 0.35 * corridor_features["entry_progress"]
            + 0.2 * corridor_features["traverse_progress"]
            + 0.18 * feedback["slot_order_readiness"]
            + 0.12 * feedback["lateral_order_readiness"]
            + 0.08 * feedback["crowding_mean"]
        )
        release_need = clip01(
            0.2 * corridor_features["entry_progress"]
            + 0.35 * corridor_features["traverse_progress"]
            + 0.25 * corridor_features["exit_progress"]
            + 0.2 * corridor_features["post_bottleneck_opening"]
            + 0.28 * feedback["release_ready"]
            + 0.12 * feedback["forward_room_mean"]
        )
        recovery_need = clip01(
            form_anisotropy_mean
            * (0.55 + 0.45 * corridor_features["teammate_pressure"])
            * (0.55 + 0.45 * open_space)
        )

        encircle_logit = (
            1.25 * open_space
            + 0.65 * (1.0 - speed_norm)
            + 0.2 * (1.0 - form_anisotropy_mean)
            - 1.05 * corridor_confidence
            - 0.35 * corridor_features["entry_progress"]
            - 0.55 * release_need
            - 0.5 * feedback["release_ready"]
            - 0.25 * feedback["slot_order_readiness"]
        )
        corridor_logit = (
            1.25 * corridor_confidence
            + 0.65 * env_anisotropy_mean
            + 0.45 * corridor_features["obstacle_pressure"]
            + 0.25 * corridor_features["entry_progress"]
            + 0.2 * corridor_features["traverse_progress"]
            + 0.2 * (1.0 - form_anisotropy_mean)
            + 0.2 * feedback["crowding_peak"]
            + 0.12 * feedback["slot_order_readiness"]
            - 0.5 * release_need
        )
        queue_logit = (
            1.2 * queue_activation
            + 0.55 * speed_norm
            + 0.35 * corridor_features["teammate_pressure"]
            + 0.2 * corridor_commitment
            + 0.25 * release_need
            + 0.1 * (1.0 - corridor_features["obstacle_pressure"])
            + 0.35 * feedback["release_ready"]
            + 0.15 * feedback["forward_room_mean"]
            + 0.1 * feedback["lateral_order_readiness"]
        )
        recover_logit = (
            0.9 * recovery_need
            + 0.45 * corridor_features["teammate_pressure"]
            + 0.35 * release_need
            + 0.25 * form_anisotropy_mean
            + 0.25 * feedback["crowding_peak"] * (1.0 - feedback["release_ready"])
        )
        mode_logits = np.array([encircle_logit, corridor_logit, queue_logit, recover_logit], dtype=float)
        mode_weights = self._softmax(mode_logits, temperature=self.params.polar_mode_temperature)

        return {
            **structure,
            **corridor_features,
            "corridor_activation": corridor_activation,
            "queue_activation": queue_activation,
            "queue_mode": queue_mode,
            "corridor_confidence": corridor_confidence,
            "corridor_commitment": corridor_commitment,
            "release_need": release_need,
            "xi_feedback_crowding_mean": feedback["crowding_mean"],
            "xi_feedback_crowding_peak": feedback["crowding_peak"],
            "xi_feedback_slot_order_readiness": feedback["slot_order_readiness"],
            "xi_feedback_lateral_order_readiness": feedback["lateral_order_readiness"],
            "xi_feedback_forward_room_mean": feedback["forward_room_mean"],
            "xi_feedback_obstacle_block_mean": feedback["obstacle_block_mean"],
            "xi_feedback_release_ready": feedback["release_ready"],
            "env_pressure_mean": env_pressure_mean,
            "env_pressure_peak": env_pressure_peak,
            "env_anisotropy_mean": env_anisotropy_mean,
            "form_anisotropy_mean": form_anisotropy_mean,
            "mode_logits": mode_logits,
            "mode_weights": mode_weights,
        }

    def _local_spacing_risk(self, idx: int, positions: np.ndarray, neighbors: list[int]) -> float:
        if not neighbors or self.params.d_agent_safe <= EPS:
            return 0.0
        min_distance = min(norm(positions[idx] - positions[j]) for j in neighbors)
        return clip01((self.params.d_agent_safe - min_distance) / max(self.params.d_agent_safe, EPS))

    def _self_localization_state(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        axis: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        axis = unit(axis)
        if norm(axis) <= EPS:
            axis = np.array([1.0, 0.0], dtype=float)
        perp_axis = rotate90(axis)
        rel = positions - center[None, :]
        parallel_coords = rel @ axis
        lateral_coords = rel @ perp_axis
        order = np.argsort(-parallel_coords, kind="stable")
        rank = int(np.where(order == idx)[0][0]) if len(order) > 1 else 0
        front_rank = 1.0 - 2.0 * rank / max(len(order) - 1, 1)
        lateral_scale = max(float(np.max(np.abs(lateral_coords))), 0.5 * self.params.r0, EPS)
        l_i = float(np.clip(lateral_coords[idx] / lateral_scale, -1.0, 1.0))

        shell = 1.0
        if neighbors:
            directions = []
            for j in neighbors:
                rel_ij = positions[j] - positions[idx]
                if norm(rel_ij) > EPS:
                    directions.append(unit(rel_ij))
            if directions:
                shell = clip01(norm(np.mean(np.array(directions), axis=0)))

        ahead_gap = float("inf")
        behind_gap = float("inf")
        for j in neighbors:
            delta = float(parallel_coords[j] - parallel_coords[idx])
            lateral_delta = abs(float(lateral_coords[j] - lateral_coords[idx]))
            if lateral_delta > 0.85 * self.params.r0:
                continue
            if delta > 0.0:
                ahead_gap = min(ahead_gap, delta)
            elif delta < 0.0:
                behind_gap = min(behind_gap, -delta)
        ahead_gap = ahead_gap if np.isfinite(ahead_gap) else 2.0 * self.params.geo_queue_spacing
        behind_gap = behind_gap if np.isfinite(behind_gap) else 2.0 * self.params.geo_queue_spacing
        gap_bias = float(np.clip((ahead_gap - behind_gap) / max(ahead_gap + behind_gap, EPS), -1.0, 1.0))

        forward_pressure = 0.0
        backward_pressure = 0.0
        for obstacle in self.params_obstacles:
            d, _normal = self._obstacle_distance(positions[idx], obstacle)
            xi = clip01((self.params.d_s - d) / max(self.params.d_s - self.params.d_obs_safe, EPS))
            if xi <= EPS:
                continue
            obs_dir = obstacle.center - positions[idx]
            forwardness = float(obs_dir @ axis)
            if forwardness >= 0.0:
                forward_pressure += xi
            else:
                backward_pressure += xi
        pressure_bias = float(
            np.clip(
                (backward_pressure - forward_pressure) / max(forward_pressure + backward_pressure + 1e-6, 1.0),
                -1.0,
                1.0,
            )
        )
        clearance_bias = float(np.clip(0.65 * gap_bias + 0.35 * pressure_bias, -1.0, 1.0))
        spacing_risk = self._local_spacing_risk(idx, positions, neighbors)
        crowding = clip01(0.7 * spacing_risk + 0.3 * clip01(-gap_bias))
        return np.array([front_rank, l_i, shell, clearance_bias, crowding], dtype=float)

    def _local_role_state(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        global_phase: dict[str, np.ndarray | float],
    ) -> np.ndarray:
        axis = np.asarray(global_phase["axis"], dtype=float)
        perp_axis = np.asarray(global_phase["perp_axis"], dtype=float)
        parallel_coords = np.asarray(global_phase["parallel_coords"], dtype=float)
        lateral_coords = np.asarray(global_phase["lateral_coords"], dtype=float)
        desired_parallel = np.asarray(global_phase["desired_parallel"], dtype=float)
        slot_error = float(
            np.clip(
                (desired_parallel[idx] - parallel_coords[idx]) / max(self.params.geo_queue_spacing, EPS),
                -1.0,
                1.0,
            )
        )
        lateral_scale = max(float(np.max(np.abs(lateral_coords))), 0.5 * self.params.r0, EPS)
        lateral_offset = float(np.clip(lateral_coords[idx] / lateral_scale, -1.0, 1.0))

        ahead_gap = float("inf")
        behind_gap = float("inf")
        for j in neighbors:
            delta = float(parallel_coords[j] - parallel_coords[idx])
            lateral_delta = abs(float(lateral_coords[j] - lateral_coords[idx]))
            if lateral_delta > 0.85 * self.params.r0:
                continue
            if delta > 0.0:
                ahead_gap = min(ahead_gap, delta)
            elif delta < 0.0:
                behind_gap = min(behind_gap, -delta)
        ahead_gap = ahead_gap if np.isfinite(ahead_gap) else 2.0 * self.params.geo_queue_spacing
        behind_gap = behind_gap if np.isfinite(behind_gap) else 2.0 * self.params.geo_queue_spacing
        gap_balance = float(np.clip((ahead_gap - behind_gap) / max(ahead_gap + behind_gap, EPS), -1.0, 1.0))

        forward_pressure = 0.0
        backward_pressure = 0.0
        for obstacle in self.params_obstacles:
            d, _normal = self._obstacle_distance(positions[idx], obstacle)
            xi = clip01((self.params.d_s - d) / max(self.params.d_s - self.params.d_obs_safe, EPS))
            if xi <= EPS:
                continue
            obs_dir = obstacle.center - positions[idx]
            forwardness = float(obs_dir @ axis)
            if forwardness >= 0.0:
                forward_pressure += xi
            else:
                backward_pressure += xi
        obs_bias = float(
            np.clip(
                (backward_pressure - forward_pressure) / max(forward_pressure + backward_pressure + 1e-6, 1.0),
                -1.0,
                1.0,
            )
        )
        spacing_risk = self._local_spacing_risk(idx, positions, neighbors)
        crowding = clip01(0.75 * spacing_risk + 0.25 * clip01(-gap_balance))
        return np.array([slot_error, lateral_offset, gap_balance, obs_bias, crowding], dtype=float)

    def _decision_coordinates(
        self,
        omega_i: float,
        psi_i: float,
        xi: np.ndarray,
        corridor_activation: float,
        queue_activation: float,
    ) -> tuple[np.ndarray, float]:
        # A compact, scene-agnostic decision coordinate system.
        # z = [progress, caution, reshape, spacing]
        # coverage_aux is a Psi-driven geometric helper, not a full decision axis.
        s_i, l_i, b_i, c_i, q_i = [float(value) for value in xi]
        backness = clip01(0.5 * (1.0 - s_i))
        frontness = clip01(0.5 * (s_i + 1.0))
        clearance = clip01(0.5 * (c_i + 1.0))
        yield_need = clip01(0.5 * (1.0 - c_i))
        lateral_offset = abs(l_i)
        shell_break = clip01(1.0 - b_i)
        pressure = clip01(omega_i)
        psi_deficit = clip01(1.0 - psi_i)
        queue_need = clip01(max(corridor_activation, queue_activation, pressure))
        open_space = clip01(1.0 - pressure)

        progress = clip01((1.0 - q_i) * (0.7 * clearance + 0.3 * backness))
        caution = clip01(0.5 * pressure + 0.3 * q_i + 0.2 * yield_need)
        reshape = clip01(0.55 * lateral_offset + 0.45 * shell_break)
        spacing = clip01(0.6 * q_i + 0.25 * yield_need + 0.15 * shell_break)
        coverage_aux = clip01((1.0 - queue_need) * (0.65 * psi_deficit + 0.2 * open_space + 0.15 * (1.0 - frontness)))
        return np.array([progress, caution, reshape, spacing], dtype=float), coverage_aux

    def _local_role_decision_coordinates(
        self,
        omega_i: float,
        psi_i: float,
        xi: np.ndarray,
        global_phase: dict[str, np.ndarray | float],
    ) -> tuple[np.ndarray, float]:
        slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi]
        forward_slot = clip01(0.5 * (slot_error + 1.0))
        forward_room = clip01(0.5 * (gap_balance + 1.0))
        yield_need = clip01(0.5 * (1.0 - gap_balance))
        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
        obstacle_block = clip01(0.5 * (1.0 - obs_bias))
        corridor_weight = float(global_phase["mode_weights"][1])
        queue_weight = float(global_phase["mode_weights"][2])
        recover_weight = float(global_phase["mode_weights"][3])
        encircle_weight = float(global_phase["mode_weights"][0])
        pressure = clip01(omega_i)
        psi_deficit = clip01(1.0 - psi_i)
        phase_tight = clip01(
            max(
                float(global_phase["corridor_activation"]),
                float(global_phase["queue_activation"]),
                float(global_phase["corridor_confidence"]),
            )
        )
        corridor_commitment = clip01(float(global_phase.get("corridor_commitment", 0.0)))
        release_need = clip01(float(global_phase.get("release_need", 0.0)))
        corridor_release = clip01(max(corridor_commitment, release_need))
        obstacle_lead = clip01(0.5 * (1.0 - obs_bias))
        crowding_relief = clip01(1.0 - 0.45 * corridor_release)
        caution_relief = clip01(1.0 - 0.35 * corridor_release)

        progress = clip01(
            (1.0 - crowding)
            * (
                0.24 * forward_room
                + 0.24 * forward_slot
                + 0.14 * obstacle_clearance
                + 0.22 * corridor_release
                + 0.08 * corridor_commitment
                + 0.08 * corridor_weight * obstacle_lead
            )
        )
        caution = clip01(
            0.4 * pressure * caution_relief
            + 0.2 * crowding * caution_relief
            + 0.14 * obstacle_block
            + 0.12 * corridor_weight * (1.0 - corridor_release)
        )
        reshape = clip01(0.45 * abs(lateral_offset) + 0.25 * abs(slot_error) + 0.3 * recover_weight)
        spacing = clip01(
            0.35 * crowding * crowding_relief
            + 0.2 * yield_need
            + 0.12 * queue_weight
            + 0.08 * corridor_weight * (1.0 - corridor_release)
            + 0.08 * obstacle_block * (1.0 - corridor_release)
        )
        coverage_aux = clip01(
            (1.0 - phase_tight)
            * (0.55 * psi_deficit + 0.25 * encircle_weight + 0.2 * (1.0 - abs(lateral_offset)))
        )
        return np.array([progress, caution, reshape, spacing], dtype=float), coverage_aux

    def _decision_module_scales(self, z: np.ndarray, coverage_aux: float) -> tuple[float, float, float, float, float, float]:
        progress, caution, reshape, spacing = [float(value) for value in z]
        nav_scale = float(np.clip(0.75 + self.params.xi_nav_gain * progress - 0.25 * caution - 0.3 * spacing, 0.4, 1.15))
        safe_scale = float(1.0 + 0.35 * spacing + 0.15 * caution)
        directional_scale = float(np.clip(0.8 + 0.45 * reshape + 0.1 * caution, 0.7, 1.35))
        guard_scale = float(1.0 + self.params.xi_guard_gain * spacing)
        enc_scale = float(np.clip(0.95 + 0.15 * progress - 0.35 * spacing - 0.2 * caution + 0.08 * coverage_aux, 0.45, 1.1))
        angle_scale = float(np.clip(0.9 + 0.2 * reshape - 0.25 * caution + 0.3 * coverage_aux, 0.45, 1.15))
        return nav_scale, safe_scale, directional_scale, guard_scale, enc_scale, angle_scale

    def _energy_flow_force(
        self,
        nav: np.ndarray,
        rep: np.ndarray,
        safe: np.ndarray,
        topo: np.ndarray,
        enc: np.ndarray,
        ang: np.ndarray,
        omega_i: float,
        rho_i: float,
        queue_mode: float,
    ) -> np.ndarray:
        rep_weight = self.params.energy_rep_weight * (1.0 + 0.9 * omega_i)
        safe_weight = self.params.energy_safe_weight * (1.0 + 0.5 * queue_mode)
        topo_weight = self.params.energy_topo_weight * (0.8 + 0.4 * rho_i)
        enc_weight = self.params.energy_enc_weight * (1.0 - 0.85 * queue_mode)
        angle_weight = self.params.energy_angle_weight * (1.0 - 0.5 * queue_mode)
        return (
            self._blend_force(nav, self.params.energy_nav_weight)
            + self._blend_force(rep, rep_weight)
            + self._blend_force(safe, safe_weight)
            + self._blend_force(topo, topo_weight)
            + self._blend_force(enc, enc_weight)
            + self._blend_force(ang, angle_weight)
        )

    def _polar_kernel_force(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        axis: np.ndarray,
        activation: float,
    ) -> np.ndarray:
        if not self.method.use_polar_kernel or activation <= EPS:
            return np.zeros(2)
        axis = unit(axis)
        if norm(axis) <= EPS:
            return np.zeros(2)
        perp_axis = rotate90(axis)
        total = np.zeros(2)
        for j in neighbors:
            rel = positions[j] - positions[idx]
            parallel = float(rel @ axis)
            lateral = float(rel @ perp_axis)
            direction_parallel = np.sign(parallel) if abs(parallel) > 1e-6 else 0.0
            parallel_mag = abs(parallel)
            parallel_term = activation * self.params.polar_parallel_gain * np.tanh((parallel_mag - self.params.geo_queue_spacing) / max(self.params.geo_queue_spacing, EPS)) * direction_parallel * axis
            lateral_term = -activation * self.params.polar_lateral_gain * np.tanh(lateral / max(self.params.polar_lateral_band, EPS)) * perp_axis
            total += parallel_term + lateral_term
        return total

    def _queue_spacing_guard_force(
        self,
        positions: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
        queue_mode: float,
    ) -> np.ndarray:
        if queue_mode <= EPS or not (self.method.use_energy_flow and self.method.use_directional_topology):
            return np.zeros_like(positions)
        axis = self._directional_axis(positions, target, predicted_target)
        axis = unit(axis)
        if norm(axis) <= EPS:
            return np.zeros_like(positions)
        center = np.mean(positions, axis=0)
        parallel_coords = np.array([float((position - center) @ axis) for position in positions])
        order = self.prev_queue_order if len(self.prev_queue_order) == len(positions) else np.argsort(-parallel_coords, kind="stable")
        guard_force = np.zeros_like(positions)
        desired_gap = self.params.hybrid_guard_spacing
        for lead_idx, follow_idx in zip(order[:-1], order[1:]):
            gap = parallel_coords[lead_idx] - parallel_coords[follow_idx]
            if gap >= desired_gap:
                continue
            correction = queue_mode * self.params.hybrid_guard_gain * np.tanh((desired_gap - gap) / max(desired_gap, EPS)) * axis
            guard_force[lead_idx] += correction
            guard_force[follow_idx] -= correction
        return guard_force

    def _magnetic_corridor_force(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
    ) -> np.ndarray:
        if not self.method.use_magnetic_corridor_adapter or global_phase is None:
            return np.zeros(2)
        stage_strength = self._corridor_low_level_stage_strength(global_phase)
        if stage_strength <= EPS:
            return np.zeros(2)

        axis = unit(np.asarray(global_phase["axis"], dtype=float))
        if norm(axis) <= EPS:
            return np.zeros(2)
        desired_parallel = np.asarray(global_phase["desired_parallel"], dtype=float)
        parallel_coords = np.asarray(global_phase["parallel_coords"], dtype=float)
        lateral_coords = np.asarray(global_phase["lateral_coords"], dtype=float)
        mode_weights = np.asarray(global_phase["mode_weights"], dtype=float)
        position = positions[idx]
        desired_gap = max(self.params.geo_queue_spacing, EPS)
        lane_band = max(0.9 * self.params.r0, EPS)

        slot_error = float((desired_parallel[idx] - parallel_coords[idx]) / desired_gap)
        lateral_error = float(lateral_coords[idx] / max(0.55 * self.params.r0, EPS))
        alignment_ready = clip01(1.0 - 0.5 * abs(float(xi_state[0])) - 0.5 * abs(float(xi_state[1])))
        crowding = clip01(float(xi_state[4]))

        slot_force = stage_strength * self.params.magnetic_slot_gain * np.tanh(slot_error) * axis
        lane_force = -stage_strength * self.params.magnetic_lane_gain * np.tanh(lateral_error) * perp_axis

        spacing_force = np.zeros(2, dtype=float)
        lateral_rep_force = np.zeros(2, dtype=float)
        for j in neighbors:
            rel = positions[j] - position
            parallel = float(rel @ axis)
            lateral = float(rel @ perp_axis)
            gap = abs(parallel)
            if gap <= EPS:
                continue
            if abs(lateral) <= lane_band:
                spacing_force += (
                    stage_strength
                    * self.params.magnetic_spacing_gain
                    * np.tanh((gap - desired_gap) / desired_gap)
                    * np.sign(parallel)
                    * axis
                )
            if abs(parallel) <= 0.75 * desired_gap and abs(lateral) <= 1.2 * lane_band:
                lateral_rep_force += (
                    -stage_strength
                    * self.params.magnetic_lateral_rep_gain
                    * np.tanh(lateral / lane_band)
                    * perp_axis
                )

        obstacle_force = np.zeros(2, dtype=float)
        tangent_force = np.zeros(2, dtype=float)
        corridor_weight = float(mode_weights[1]) if mode_weights.shape == (4,) else 0.0
        for obstacle in self.params_obstacles:
            d, normal = self._obstacle_distance(position, obstacle)
            xi = clip01((self.params.d_s - d) / max(self.params.d_s - self.params.d_obs_safe, EPS))
            if xi <= EPS:
                continue
            obstacle_force += stage_strength * self.params.magnetic_obstacle_gain * (xi ** 2) * normal
            tangent = rotate90(normal)
            if float(tangent @ axis) < 0.0:
                tangent = -tangent
            tangent_force += (
                stage_strength
                * corridor_weight
                * self.params.magnetic_tangent_gain
                * xi
                * clip01(1.0 - abs(float(normal @ axis)))
                * tangent
            )

        release_signal = clip01(
            max(
                float(global_phase.get("corridor_commitment", 0.0)),
                float(global_phase.get("release_need", 0.0)),
                float(global_phase.get("xi_feedback_release_ready", 0.0)),
            )
        )
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        release_force = (
            stage_strength
            * self.params.magnetic_release_gain
            * release_signal
            * (0.35 + 0.65 * forward_free_norm)
            * (0.45 + 0.55 * alignment_ready)
            * (1.0 - 0.35 * crowding)
            * axis
        )
        return slot_force + lane_force + spacing_force + lateral_rep_force + obstacle_force + tangent_force + release_force

    def _xi_magnetic_corridor_force(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
    ) -> np.ndarray:
        if not self.method.use_xi_magnetic_polarity or global_phase is None:
            return np.zeros(2)
        stage_strength = self._corridor_low_level_stage_strength(global_phase)
        if stage_strength <= EPS:
            return np.zeros(2)

        axis = unit(np.asarray(global_phase["axis"], dtype=float))
        if norm(axis) <= EPS:
            return np.zeros(2)
        perp_axis = np.asarray(global_phase["perp_axis"], dtype=float)
        desired_parallel = np.asarray(global_phase["desired_parallel"], dtype=float)
        parallel_coords = np.asarray(global_phase["parallel_coords"], dtype=float)
        lateral_coords = np.asarray(global_phase["lateral_coords"], dtype=float)
        mode_weights = np.asarray(global_phase["mode_weights"], dtype=float)
        position = positions[idx]
        desired_gap = max(self.params.geo_queue_spacing, EPS)
        lane_band = max(0.9 * self.params.r0, EPS)

        slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_state]
        forward_slot = clip01(0.5 * (slot_error + 1.0))
        yield_need = clip01(0.5 * (1.0 - gap_balance))
        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
        obstacle_block = clip01(0.5 * (1.0 - obs_bias))
        xi_stability = clip01(1.0 - 0.4 * abs(slot_error) - 0.35 * abs(lateral_offset) - 0.25 * crowding)
        local_axis, local_perp = self._xi_polarity_frame(idx, global_phase, xi_state)

        slot_gain = self.params.magnetic_slot_gain * (0.7 + 0.6 * abs(slot_error) + 0.35 * forward_slot)
        lane_gain = self.params.magnetic_lane_gain * (0.6 + 0.8 * abs(lateral_offset) + 0.25 * yield_need)
        spacing_gain = self.params.magnetic_spacing_gain * (0.8 + 0.9 * crowding + 0.4 * yield_need)
        lateral_rep_gain = self.params.magnetic_lateral_rep_gain * (0.8 + 0.6 * abs(lateral_offset) + 0.4 * crowding)
        obstacle_gain = self.params.magnetic_obstacle_gain * (0.75 + 0.6 * obstacle_block)
        tangent_gain = self.params.magnetic_tangent_gain * (0.55 + 0.75 * obstacle_block + 0.25 * yield_need)

        slot_error_local = float((desired_parallel[idx] - parallel_coords[idx]) / desired_gap)
        lateral_error_local = float(lateral_coords[idx] / max(0.55 * self.params.r0, EPS))
        slot_force = stage_strength * slot_gain * np.tanh(slot_error_local) * local_axis
        lane_force = -stage_strength * lane_gain * np.tanh(lateral_error_local) * local_perp

        spacing_force = np.zeros(2, dtype=float)
        lateral_rep_force = np.zeros(2, dtype=float)
        for j in neighbors:
            rel = positions[j] - position
            parallel = float(rel @ local_axis)
            lateral = float(rel @ local_perp)
            gap = abs(parallel)
            if gap <= EPS:
                continue
            if abs(lateral) <= lane_band:
                asym = 1.0 + 0.35 * yield_need if parallel > 0.0 else 1.0 - 0.15 * yield_need
                spacing_force += (
                    stage_strength
                    * spacing_gain
                    * asym
                    * np.tanh((gap - desired_gap * (1.0 + 0.25 * crowding)) / desired_gap)
                    * np.sign(parallel)
                    * local_axis
                )
            if abs(parallel) <= 0.75 * desired_gap and abs(lateral) <= 1.2 * lane_band:
                lateral_rep_force += (
                    -stage_strength
                    * lateral_rep_gain
                    * np.tanh(lateral / lane_band)
                    * local_perp
                )

        obstacle_force = np.zeros(2, dtype=float)
        tangent_force = np.zeros(2, dtype=float)
        corridor_weight = float(mode_weights[1]) if mode_weights.shape == (4,) else 0.0
        for obstacle in self.params_obstacles:
            d, normal = self._obstacle_distance(position, obstacle)
            xi = clip01((self.params.d_s - d) / max(self.params.d_s - self.params.d_obs_safe, EPS))
            if xi <= EPS:
                continue
            obstacle_force += stage_strength * obstacle_gain * (xi ** 2) * normal
            tangent = rotate90(normal)
            if float(tangent @ local_axis) < 0.0:
                tangent = -tangent
            tangent_force += (
                stage_strength
                * corridor_weight
                * tangent_gain
                * xi
                * clip01(1.0 - abs(float(normal @ local_axis)))
                * tangent
            )

        release_signal = clip01(
            max(
                float(global_phase.get("corridor_commitment", 0.0)),
                float(global_phase.get("release_need", 0.0)),
                float(global_phase.get("xi_feedback_release_ready", 0.0)),
            )
        )
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        release_force = (
            stage_strength
            * self.params.magnetic_release_gain
            * (0.4 + 0.6 * xi_stability + 0.35 * forward_slot)
            * release_signal
            * (0.35 + 0.65 * forward_free_norm)
            * (1.0 - 0.45 * crowding)
            * local_axis
        )
        return slot_force + lane_force + spacing_force + lateral_rep_force + obstacle_force + tangent_force + release_force

    def _xi_polarity_frame(
        self,
        idx: int,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.method.use_xi_polarity_axis_field or global_phase is None:
            axis = np.array([1.0, 0.0], dtype=float)
            return axis, rotate90(axis)
        axis = unit(np.asarray(global_phase["axis"], dtype=float))
        if norm(axis) <= EPS:
            axis = np.array([1.0, 0.0], dtype=float)
        perp_axis = np.asarray(global_phase["perp_axis"], dtype=float)
        slot_error, lateral_offset, gap_balance, obs_bias, _crowding = [float(value) for value in xi_state]
        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
        obstacle_block = clip01(0.5 * (1.0 - obs_bias))
        obstacle_bias = obstacle_clearance - obstacle_block
        polarity_axis = unit(
            axis
            - self.params.magnetic_polarity_lateral_gain * lateral_offset * perp_axis
            + self.params.magnetic_polarity_gap_gain * gap_balance * axis
            + self.params.magnetic_polarity_obstacle_gain * obstacle_bias * perp_axis
        )
        if norm(polarity_axis) <= EPS:
            polarity_axis = axis
        return polarity_axis, rotate90(polarity_axis)

    def _xi_temporal_magnetic_corridor_force(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
    ) -> np.ndarray:
        if not self.method.use_temporal_magnetic_release or global_phase is None:
            return np.zeros(2)
        stage_strength = self._corridor_low_level_stage_strength(global_phase)
        if stage_strength <= EPS:
            return np.zeros(2)

        axis = unit(np.asarray(global_phase["axis"], dtype=float))
        if norm(axis) <= EPS:
            return np.zeros(2)
        perp_axis = np.asarray(global_phase["perp_axis"], dtype=float)
        queue_order = np.asarray(global_phase["queue_order"], dtype=int)
        parallel_coords = np.asarray(global_phase["parallel_coords"], dtype=float)
        lateral_coords = np.asarray(global_phase["lateral_coords"], dtype=float)
        desired_parallel = np.asarray(global_phase["desired_parallel"], dtype=float)
        mode_weights = np.asarray(global_phase["mode_weights"], dtype=float)
        position = positions[idx]
        desired_gap = max(self.params.geo_queue_spacing, EPS)
        lane_band = max(0.9 * self.params.r0, EPS)

        rank = int(np.where(queue_order == idx)[0][0]) if len(queue_order) else 0
        n_agents = max(len(positions) - 1, 1)
        priority = clip01(1.0 - rank / n_agents)
        slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_state]
        xi_stability = clip01(1.0 - 0.35 * abs(slot_error) - 0.3 * abs(lateral_offset) - 0.35 * crowding)
        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
        obstacle_block = clip01(0.5 * (1.0 - obs_bias))

        local_axis = unit(axis - 0.24 * lateral_offset * perp_axis + 0.14 * (obstacle_clearance - obstacle_block) * perp_axis)
        if norm(local_axis) <= EPS:
            local_axis = axis
        local_perp = rotate90(local_axis)

        front_idx = int(queue_order[rank - 1]) if rank > 0 else None
        back_idx = int(queue_order[rank + 1]) if rank + 1 < len(queue_order) else None
        front_gap = float("inf")
        closing_speed = 0.0
        predecessor_gate = 1.0
        if front_idx is not None:
            front_gap = parallel_coords[front_idx] - parallel_coords[idx]
            self_forward = max(float(self.prev_output_vel[idx] @ local_axis), 0.0)
            front_forward = max(float(self.prev_output_vel[front_idx] @ local_axis), 0.0)
            closing_speed = max(self_forward - front_forward, 0.0)
            predecessor_gate = float(self.magnetic_release_prev[front_idx])
        front_gap = front_gap if np.isfinite(front_gap) else 10.0 * desired_gap

        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        corridor_drive = clip01(
            max(
                float(global_phase.get("corridor_commitment", 0.0)),
                float(global_phase.get("entry_progress", 0.0)),
                float(global_phase.get("traverse_progress", 0.0)),
            )
        )
        release_seed = clip01(
            0.4 * corridor_drive
            + 0.25 * float(global_phase.get("xi_feedback_release_ready", 0.0))
            + 0.2 * forward_free_norm
            + 0.15 * xi_stability
        )
        safe_gap = desired_gap * (1.0 + 0.2 * crowding) + self.params.magnetic_headway_time * closing_speed
        gap_margin = clip01((front_gap - safe_gap + desired_gap) / max(2.0 * desired_gap, EPS))

        if front_idx is None:
            gate_score = clip01(release_seed + 0.25 * priority + 0.15 * float(global_phase.get("entry_progress", 0.0)))
        else:
            gate_score = clip01(0.35 * release_seed + 0.3 * predecessor_gate + 0.25 * gap_margin + 0.1 * priority)

        prev_gate = float(self.magnetic_release_prev[idx])
        if prev_gate >= 0.5:
            gate = 1.0 if gate_score >= self.params.magnetic_release_off else 0.0
        else:
            gate = 1.0 if gate_score >= self.params.magnetic_release_on else 0.0
        self.magnetic_release_state[idx] = gate

        slot_force = stage_strength * self.params.magnetic_slot_gain * (0.6 + 0.45 * priority) * np.tanh((desired_parallel[idx] - parallel_coords[idx]) / desired_gap) * local_axis
        lane_force = -stage_strength * self.params.magnetic_lane_gain * (0.6 + 0.6 * abs(lateral_offset) + 0.25 * (1.0 - gate)) * np.tanh(lateral_coords[idx] / max(0.55 * self.params.r0, EPS)) * local_perp

        headway_force = np.zeros(2, dtype=float)
        if front_idx is not None:
            headway_error = safe_gap - front_gap
            if headway_error > 0.0:
                headway_force += -stage_strength * self.params.magnetic_headway_gain * np.tanh(headway_error / desired_gap) * local_axis
            elif gate >= 0.5 and predecessor_gate >= 0.5:
                headway_force += 0.25 * stage_strength * self.params.magnetic_headway_gain * gap_margin * local_axis

        back_pressure_force = np.zeros(2, dtype=float)
        if back_idx is not None:
            back_gap = parallel_coords[idx] - parallel_coords[back_idx]
            if back_gap < 0.85 * desired_gap and gate >= 0.5:
                back_pressure_force += 0.12 * stage_strength * (0.85 - back_gap / desired_gap) * local_axis

        spacing_force = np.zeros(2, dtype=float)
        lateral_rep_force = np.zeros(2, dtype=float)
        for j in neighbors:
            rel = positions[j] - position
            parallel = float(rel @ local_axis)
            lateral = float(rel @ local_perp)
            gap = abs(parallel)
            if gap <= EPS:
                continue
            if abs(lateral) <= lane_band:
                asym = 1.0 + 0.55 * (1.0 - priority) if parallel > 0.0 else 0.85 + 0.2 * priority
                spacing_force += (
                    stage_strength
                    * self.params.magnetic_spacing_gain
                    * asym
                    * np.tanh((gap - desired_gap * (1.0 + 0.2 * crowding)) / desired_gap)
                    * np.sign(parallel)
                    * local_axis
                )
            if abs(parallel) <= 0.75 * desired_gap and abs(lateral) <= 1.2 * lane_band:
                lateral_rep_force += (
                    -stage_strength
                    * self.params.magnetic_lateral_rep_gain
                    * (0.85 + 0.4 * crowding + 0.15 * (1.0 - gate))
                    * np.tanh(lateral / lane_band)
                    * local_perp
                )

        obstacle_force = np.zeros(2, dtype=float)
        tangent_force = np.zeros(2, dtype=float)
        corridor_weight = float(mode_weights[1]) if mode_weights.shape == (4,) else 0.0
        for obstacle in self.params_obstacles:
            d, normal = self._obstacle_distance(position, obstacle)
            xi = clip01((self.params.d_s - d) / max(self.params.d_s - self.params.d_obs_safe, EPS))
            if xi <= EPS:
                continue
            obstacle_force += stage_strength * self.params.magnetic_obstacle_gain * (0.8 + 0.45 * obstacle_block) * (xi ** 2) * normal
            tangent = rotate90(normal)
            if float(tangent @ local_axis) < 0.0:
                tangent = -tangent
            tangent_force += (
                stage_strength
                * corridor_weight
                * self.params.magnetic_tangent_gain
                * (0.7 + 0.35 * obstacle_block)
                * xi
                * clip01(1.0 - abs(float(normal @ local_axis)))
                * tangent
            )

        pre_release_force = stage_strength * self.params.magnetic_pre_release_gain * corridor_drive * (0.15 + 0.85 * priority) * local_axis
        release_force = stage_strength * self.params.magnetic_release_gain * gate * (0.4 + 0.6 * forward_free_norm) * (0.35 + 0.65 * xi_stability) * local_axis
        return (
            slot_force
            + lane_force
            + spacing_force
            + lateral_rep_force
            + obstacle_force
            + tangent_force
            + pre_release_force
            + release_force
            + headway_force
            + back_pressure_force
        )

    def _magnetic_drive_force(
        self,
        idx: int,
        nav: np.ndarray,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        if not self.method.use_magnetic_drive_energy or global_phase is None:
            return np.zeros(2), 0.0, 0.0
        stage_strength = self._corridor_low_level_stage_strength(global_phase)
        if stage_strength <= EPS:
            return np.zeros(2), 0.0, 0.0

        axis = unit(np.asarray(global_phase["axis"], dtype=float))
        if norm(axis) <= EPS:
            return np.zeros(2), 0.0, 0.0
        nav_axis = unit(nav)
        if norm(nav_axis) <= EPS:
            nav_axis = axis
        local_axis, _local_perp = self._xi_polarity_frame(idx, global_phase, xi_state)
        drive_axis = unit(
            self.params.magnetic_drive_nav_blend * nav_axis
            + (1.0 - self.params.magnetic_drive_nav_blend) * local_axis
        )
        if norm(drive_axis) <= EPS:
            drive_axis = local_axis

        slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_state]
        forward_slot = clip01(0.5 * (slot_error + 1.0))
        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
        xi_alignment = clip01(1.0 - 0.45 * abs(lateral_offset) - 0.35 * crowding)
        corridor_drive = clip01(
            max(
                float(global_phase.get("corridor_commitment", 0.0)),
                float(global_phase.get("entry_progress", 0.0)),
                float(global_phase.get("traverse_progress", 0.0)),
            )
        )
        release_need = clip01(float(global_phase.get("release_need", 0.0)))
        release_ready = clip01(float(global_phase.get("xi_feedback_release_ready", 0.0)))
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        approach_need = clip01(1.0 - float(global_phase.get("entry_progress", 0.0)))

        drive_gate = clip01(
            0.3 * corridor_drive
            + 0.2 * forward_slot
            + 0.2 * obstacle_clearance
            + 0.15 * xi_alignment
            + 0.15 * clip01(1.0 - crowding)
        )
        release_gate = clip01(0.45 * release_need + 0.3 * release_ready + 0.25 * forward_free_norm)
        drive_stage = stage_strength
        base_gain = self.params.magnetic_drive_gain
        release_gain = self.params.magnetic_drive_release_gain
        if self.method.use_aggressive_magnetic_soft_drive:
            queue_order = np.asarray(global_phase.get("queue_order", np.arange(1)), dtype=int)
            rank = int(np.where(queue_order == idx)[0][0]) if idx in queue_order else 0
            n_agents = max(len(queue_order) - 1, 1)
            leader_priority = clip01(1.0 - rank / n_agents)
            drive_stage = max(
                stage_strength,
                self.params.magnetic_approach_floor
                * clip01(0.45 + 0.55 * approach_need + 0.2 * leader_priority),
            )
            base_gain = self.params.magnetic_approach_gain * (1.0 + self.params.magnetic_soft_leader_gain * leader_priority)
            release_gain = self.params.magnetic_drive_release_gain * (1.0 + 0.25 * leader_priority)
        if self.method.use_high_energy_polarity_drive:
            drive_stage = max(drive_stage, self.params.magnetic_high_approach_floor)
            base_gain = max(base_gain, self.params.magnetic_high_energy_gain)
            release_gain = max(release_gain, self.params.magnetic_high_release_gain)
        if self.method.use_adaptive_xi_energy_drive:
            slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_state]
            obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
            forward_room = clip01(0.5 * (gap_balance + 1.0))
            corridor_confidence = clip01(float(global_phase.get("corridor_confidence", 0.0)))
            corridor_commitment = clip01(float(global_phase.get("corridor_commitment", 0.0)))
            release_need = clip01(float(global_phase.get("release_need", 0.0)))
            forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
            xi_alignment = clip01(1.0 - 0.45 * abs(lateral_offset) - 0.35 * abs(slot_error))
            adaptive_gate = clip01(
                0.24 * obstacle_clearance
                + 0.18 * forward_room
                + 0.18 * corridor_commitment
                + 0.14 * forward_free_norm
                + 0.14 * xi_alignment
                + 0.12 * release_need
                - 0.24 * crowding
                - 0.1 * corridor_confidence * crowding
            )
            base_gain = float(
                np.clip(
                    self.params.magnetic_adaptive_energy_floor
                    + (self.params.magnetic_adaptive_energy_ceiling - self.params.magnetic_adaptive_energy_floor) * adaptive_gate,
                    self.params.magnetic_adaptive_energy_floor,
                    self.params.magnetic_adaptive_energy_ceiling,
                )
            )
            release_gain = float(
                np.clip(
                    0.45 + (self.params.magnetic_adaptive_release_ceiling - 0.45) * adaptive_gate,
                    0.45,
                    self.params.magnetic_adaptive_release_ceiling,
                )
            )
        base_drive = drive_stage * base_gain * (0.25 + 0.75 * drive_gate) * drive_axis
        release_drive = drive_stage * release_gain * release_gate * (0.35 + 0.65 * forward_free_norm) * local_axis
        return base_drive + release_drive, norm(base_drive), norm(release_drive)

    def _soft_queue_occupancy_force(
        self,
        idx: int,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if not self.method.use_soft_queue_occupancy_rule or global_phase is None:
            return np.zeros(2, dtype=float), 1.0
        axis = unit(np.asarray(global_phase["axis"], dtype=float))
        if norm(axis) <= EPS:
            return np.zeros(2, dtype=float), 1.0
        queue_order = np.asarray(global_phase.get("queue_order", np.arange(1)), dtype=int)
        parallel_coords = np.asarray(global_phase.get("parallel_coords", np.zeros(1)), dtype=float)
        if idx not in queue_order:
            return np.zeros(2, dtype=float), 1.0
        rank = int(np.where(queue_order == idx)[0][0])
        if rank == 0:
            return np.zeros(2, dtype=float), 1.0

        front_idx = int(queue_order[rank - 1])
        front_gap = float(parallel_coords[front_idx] - parallel_coords[idx])
        desired_gap = max(self.params.geo_queue_spacing, EPS)
        slot_error, _lateral_offset, gap_balance, _obs_bias, crowding = [float(value) for value in xi_state]
        front_zone = self.params.magnetic_queue_front_zone * desired_gap
        occupancy = clip01((front_zone - front_gap) / max(front_zone, EPS))
        if occupancy <= EPS:
            return np.zeros(2, dtype=float), 1.0

        queue_repulsion = -(
            self.params.magnetic_queue_occupancy_gain
            * occupancy
            * (0.8 + 0.35 * crowding + 0.25 * clip01(0.5 * (1.0 - gap_balance)))
            * axis
        )
        drive_scale = float(
            np.clip(
                1.0 - self.params.magnetic_queue_drive_drop_gain * occupancy * (0.85 + 0.15 * clip01(1.0 - abs(slot_error))),
                0.0,
                1.0,
            )
        )
        return queue_repulsion, drive_scale

    def _soft_queue_push_force(
        self,
        idx: int,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if not self.method.use_soft_queue_push_rule or global_phase is None:
            return np.zeros(2, dtype=float), 1.0
        axis = unit(np.asarray(global_phase["axis"], dtype=float))
        if norm(axis) <= EPS:
            return np.zeros(2, dtype=float), 1.0
        queue_order = np.asarray(global_phase.get("queue_order", np.arange(1)), dtype=int)
        parallel_coords = np.asarray(global_phase.get("parallel_coords", np.zeros(1)), dtype=float)
        if idx not in queue_order:
            return np.zeros(2, dtype=float), 1.0

        rank = int(np.where(queue_order == idx)[0][0])
        desired_gap = max(self.params.geo_queue_spacing, EPS)
        push_zone = self.params.magnetic_queue_push_zone * desired_gap
        slot_error, _lateral_offset, gap_balance, _obs_bias, crowding = [float(value) for value in xi_state]

        push_force = np.zeros(2, dtype=float)
        drive_scale = 1.0
        if rank + 1 < len(queue_order):
            back_idx = int(queue_order[rank + 1])
            back_gap = float(parallel_coords[idx] - parallel_coords[back_idx])
            occupancy = clip01((push_zone - back_gap) / max(push_zone, EPS))
            if occupancy > EPS:
                push_force = (
                    self.params.magnetic_queue_push_gain
                    * occupancy
                    * (0.8 + 0.3 * clip01(1.0 - abs(slot_error)) + 0.25 * crowding)
                    * axis
                )
        if rank > 0:
            front_idx = int(queue_order[rank - 1])
            front_gap = float(parallel_coords[front_idx] - parallel_coords[idx])
            compression = clip01((desired_gap - front_gap) / max(desired_gap, EPS))
            drive_scale = float(np.clip(1.0 - 0.35 * compression * (0.75 + 0.25 * clip01(0.5 * (1.0 - gap_balance))), 0.45, 1.0))
        return push_force, drive_scale

    def _soft_leader_pull_force(
        self,
        idx: int,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if not self.method.use_soft_leader_pull_rule or global_phase is None:
            return np.zeros(2, dtype=float), 1.0
        axis = unit(np.asarray(global_phase["axis"], dtype=float))
        if norm(axis) <= EPS:
            return np.zeros(2, dtype=float), 1.0
        queue_order = np.asarray(global_phase.get("queue_order", np.arange(1)), dtype=int)
        parallel_coords = np.asarray(global_phase.get("parallel_coords", np.zeros(1)), dtype=float)
        if idx not in queue_order:
            return np.zeros(2, dtype=float), 1.0
        rank = int(np.where(queue_order == idx)[0][0])
        n_agents = max(len(queue_order) - 1, 1)
        leader_priority = clip01(1.0 - rank / n_agents)
        slot_error, _lateral_offset, gap_balance, _obs_bias, crowding = [float(value) for value in xi_state]
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        corridor_drive = clip01(
            max(
                float(global_phase.get("corridor_commitment", 0.0)),
                float(global_phase.get("entry_progress", 0.0)),
                float(global_phase.get("traverse_progress", 0.0)),
            )
        )

        if rank == 0:
            pull_force = (
                self.params.magnetic_leader_pull_gain
                * (0.35 + 0.65 * forward_free_norm)
                * (0.45 + 0.55 * corridor_drive)
                * axis
            )
            drive_scale = 1.0
            return pull_force, drive_scale

        front_idx = int(queue_order[rank - 1])
        front_gap = float(parallel_coords[front_idx] - parallel_coords[idx])
        desired_gap = max(self.params.geo_queue_spacing, EPS)
        trail_gap = clip01((front_gap - desired_gap) / max(self.params.magnetic_leader_span * desired_gap, EPS))
        pull_force = (
            self.params.magnetic_leader_trail_gain
            * leader_priority
            * trail_gap
            * (0.7 + 0.3 * clip01(1.0 - abs(slot_error)))
            * axis
        )
        drive_scale = float(
            np.clip(
                1.0 - 0.45 * clip01((desired_gap - front_gap) / desired_gap) * (0.75 + 0.25 * clip01(0.5 * (1.0 - gap_balance) + crowding)),
                0.35,
                1.0,
            )
        )
        return pull_force, drive_scale

    def _corridor_low_level_stage_strength(self, global_phase: dict[str, np.ndarray | float] | None) -> float:
        if global_phase is None:
            return 0.0
        return clip01(
            float(global_phase.get("corridor_confidence", 0.0))
            * max(
                float(global_phase.get("corridor_commitment", 0.0)),
                float(global_phase.get("xi_feedback_release_ready", 0.0)),
            )
        )

    def _structured_energy_signals(self, global_phase: dict[str, np.ndarray | float] | None) -> dict[str, float]:
        if global_phase is None:
            return {
                "scene_signal": 0.0,
                "topology_signal": 0.0,
                "store": 0.0,
                "release": 0.0,
                "loss": 0.0,
                "tank": clip01(self.transport_energy_tank / max(self.params.magnetic_tank_capacity, EPS)),
            }

        corridor_confidence = clip01(float(global_phase.get("corridor_confidence", 0.0)))
        corridor_commitment = clip01(float(global_phase.get("corridor_commitment", 0.0)))
        release_need = clip01(float(global_phase.get("release_need", 0.0)))
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        obstacle_pressure = clip01(float(global_phase.get("obstacle_pressure", 0.0)))
        teammate_pressure = clip01(float(global_phase.get("teammate_pressure", 0.0)))
        lateral_width = max(float(global_phase.get("lateral_width", 0.0)), 0.0)
        longitudinal_span = max(float(global_phase.get("longitudinal_span", 0.0)), EPS)

        feedback = self.prev_xi_feedback
        slot_order = clip01(float(feedback["slot_order_readiness"]))
        lateral_order = clip01(float(feedback["lateral_order_readiness"]))
        forward_room = clip01(float(feedback["forward_room_mean"]))
        release_ready = clip01(float(feedback["release_ready"]))
        crowding_mean = clip01(float(feedback["crowding_mean"]))
        crowding_peak = clip01(float(feedback["crowding_peak"]))
        obstacle_block = clip01(float(feedback["obstacle_block_mean"]))

        # The best transport band is neither too loose nor too compressed.
        compact_band = clip01(1.0 - 2.0 * abs(crowding_mean - 0.55))
        queue_shape = clip01(1.0 - lateral_width / max(longitudinal_span + 0.45 * self.params.r0, EPS))
        scene_signal = clip01(
            0.32 * corridor_confidence
            + 0.24 * corridor_commitment
            + 0.22 * release_need
            + 0.12 * forward_free_norm
            + 0.1 * (1.0 - obstacle_pressure)
        )
        topology_signal = clip01(
            0.24 * slot_order
            + 0.2 * lateral_order
            + 0.16 * forward_room
            + 0.15 * release_ready
            + 0.12 * compact_band
            + 0.13 * queue_shape
        )

        tank_norm = clip01(self.transport_energy_tank / max(self.params.magnetic_tank_capacity, EPS))
        store = self.params.magnetic_tank_charge_gain * scene_signal * topology_signal * compact_band * (1.0 - tank_norm)
        release = self.params.magnetic_tank_release_gain * release_need * release_ready * (0.35 + 0.65 * forward_free_norm) * tank_norm
        loss = self.params.magnetic_tank_loss_gain * (0.45 * crowding_peak + 0.3 * obstacle_block + 0.15 * obstacle_pressure + 0.1 * teammate_pressure) * tank_norm
        return {
            "scene_signal": scene_signal,
            "topology_signal": topology_signal,
            "store": store,
            "release": release,
            "loss": loss,
            "tank": tank_norm,
        }

    def _update_structured_energy_tank(self, global_phase: dict[str, np.ndarray | float] | None) -> dict[str, float]:
        signals = self._structured_energy_signals(global_phase)
        capacity = max(self.params.magnetic_tank_capacity, EPS)
        tank = float(np.clip(self.transport_energy_tank + self.params.dt * (signals["store"] - signals["release"] - signals["loss"]), 0.0, capacity))
        self.transport_energy_tank = tank
        tank_norm = clip01(tank / capacity)
        self.last_energy_scene_signal = signals["scene_signal"]
        self.last_energy_topology_signal = signals["topology_signal"]
        self.last_energy_store = signals["store"]
        self.last_energy_release = signals["release"]
        self.last_energy_loss = signals["loss"]
        return {
            **signals,
            "tank": tank_norm,
        }

    def _initial_topology_energy(self, global_phase: dict[str, np.ndarray | float] | None) -> tuple[float, float]:
        if global_phase is None:
            return 0.0, 0.0
        if not self.method.use_energy_topology_layer:
            return 0.75, 0.25
        desired_parallel = np.asarray(global_phase.get("desired_parallel", np.zeros(1)), dtype=float)
        parallel_coords = np.asarray(global_phase.get("parallel_coords", np.zeros_like(desired_parallel)), dtype=float)
        desired_gap = max(self.params.geo_queue_spacing, EPS)
        slot_error_mean = 0.0
        if len(desired_parallel) == len(parallel_coords) and len(parallel_coords) > 0:
            slot_error_mean = float(np.mean(np.clip(np.abs(desired_parallel - parallel_coords) / desired_gap, 0.0, 1.0)))
        slot_alignment = clip01(1.0 - slot_error_mean)
        obstacle_pressure = clip01(float(global_phase.get("obstacle_pressure", 0.0)))
        teammate_pressure = clip01(float(global_phase.get("teammate_pressure", 0.0)))
        lateral_width = max(float(global_phase.get("lateral_width", 0.0)), 0.0)
        longitudinal_span = max(float(global_phase.get("longitudinal_span", 0.0)), EPS)
        queue_shape = clip01(1.0 - lateral_width / max(longitudinal_span + 0.45 * self.params.r0, EPS))
        compress_potential = clip01(lateral_width / max(lateral_width + longitudinal_span + self.params.r0, EPS))
        continuity = clip01(1.0 - teammate_pressure)
        obstacle_clear = clip01(1.0 - obstacle_pressure)
        ready = clip01(
            0.32 * slot_alignment
            + 0.28 * compress_potential
            + 0.2 * continuity
            + 0.2 * obstacle_clear
            + self.params.magnetic_initial_topology_bias
        )
        debt = clip01(
            0.3 * (1.0 - slot_alignment)
            + 0.2 * teammate_pressure
            + 0.15 * obstacle_pressure
            + 0.1 * (1.0 - queue_shape)
        )
        return ready, debt

    def _update_dual_stage_energy_tanks(self, global_phase: dict[str, np.ndarray | float] | None) -> dict[str, float]:
        if global_phase is None:
            return {
                "scene_signal": 0.0,
                "topology_signal": 0.0,
                "approach_tank": 0.0,
                "transport_tank": 0.0,
                "approach_store": 0.0,
                "approach_loss": 0.0,
                "transport_store": 0.0,
                "transport_release": 0.0,
                "transport_loss": 0.0,
                "convert": 0.0,
                "initial_ready": self.initial_topology_ready,
                "initial_debt": self.initial_topology_debt,
            }

        if not self.dual_stage_energy_initialized:
            ready, debt = self._initial_topology_energy(global_phase)
            self.initial_topology_ready = ready
            self.initial_topology_debt = debt
            self.approach_energy_tank = float(
                np.clip(
                    self.params.magnetic_approach_tank_capacity * max(ready - debt, 0.0),
                    0.0,
                    self.params.magnetic_approach_tank_capacity,
                )
            )
            self.transport_energy_tank = 0.0
            self.dual_stage_energy_initialized = True

        scene_signals = self._structured_energy_signals(global_phase)
        scene_signal = float(scene_signals["scene_signal"]) if self.method.use_energy_scene_layer else 1.0
        topology_signal = float(scene_signals["topology_signal"]) if self.method.use_energy_topology_layer else 1.0
        corridor_commitment = clip01(float(global_phase.get("corridor_commitment", 0.0)))
        release_need = clip01(float(global_phase.get("release_need", 0.0)))
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        obstacle_pressure = clip01(float(global_phase.get("obstacle_pressure", 0.0)))
        teammate_pressure = clip01(float(global_phase.get("teammate_pressure", 0.0)))
        feedback = self.prev_xi_feedback
        release_ready = clip01(float(feedback["release_ready"]))
        crowding_peak = clip01(float(feedback["crowding_peak"]))
        crowding_mean = clip01(float(feedback["crowding_mean"]))
        compact_band = clip01(1.0 - 2.0 * abs(crowding_mean - 0.4))

        approach_capacity = max(self.params.magnetic_approach_tank_capacity, EPS)
        transport_capacity = max(self.params.magnetic_tank_capacity, EPS)
        approach_norm = clip01(self.approach_energy_tank / approach_capacity)
        transport_norm = clip01(self.transport_energy_tank / transport_capacity)

        pre_entry = clip01(1.0 - float(global_phase.get("entry_progress", 0.0)))
        approach_store = self.params.magnetic_approach_tank_charge_gain * scene_signal * topology_signal * pre_entry * compact_band * (1.0 - approach_norm)
        convert = self.params.magnetic_transport_convert_gain * corridor_commitment * release_ready * self.approach_energy_tank
        approach_loss = self.params.magnetic_approach_tank_loss_gain * (0.45 * obstacle_pressure + 0.35 * teammate_pressure + 0.2 * crowding_peak) * self.approach_energy_tank
        self.approach_energy_tank = float(
            np.clip(
                self.approach_energy_tank + self.params.dt * (approach_store - convert - approach_loss),
                0.0,
                approach_capacity,
            )
        )

        transport_store = convert + self.params.magnetic_transport_charge_gain * scene_signal * topology_signal * corridor_commitment * (1.0 - transport_norm)
        transport_release = self.params.magnetic_transport_release_gain * release_need * release_ready * (0.35 + 0.65 * forward_free_norm) * self.transport_energy_tank
        transport_loss = self.params.magnetic_transport_loss_gain * (0.5 * crowding_peak + 0.25 * obstacle_pressure + 0.25 * teammate_pressure) * self.transport_energy_tank
        self.transport_energy_tank = float(
            np.clip(
                self.transport_energy_tank + self.params.dt * (transport_store - transport_release - transport_loss),
                0.0,
                transport_capacity,
            )
        )

        self.last_energy_scene_signal = scene_signal
        self.last_energy_topology_signal = topology_signal
        self.last_energy_store = transport_store
        self.last_energy_release = transport_release
        self.last_energy_loss = transport_loss
        return {
            "scene_signal": scene_signal,
            "topology_signal": topology_signal,
            "approach_tank": clip01(self.approach_energy_tank / approach_capacity),
            "transport_tank": clip01(self.transport_energy_tank / transport_capacity),
            "approach_store": approach_store,
            "approach_loss": approach_loss,
            "transport_store": transport_store,
            "transport_release": transport_release,
            "transport_loss": transport_loss,
            "convert": convert,
            "initial_ready": self.initial_topology_ready,
            "initial_debt": self.initial_topology_debt,
        }

    def _dual_stage_agent_budget(
        self,
        idx: int,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
        queue_drive_scale: float,
        propagated_entry_budget: float = 0.0,
    ) -> tuple[float, float, float]:
        if not self.method.use_dual_stage_energy_tank or global_phase is None:
            return 0.0, 1.0, 1.0
        queue_order = np.asarray(global_phase.get("queue_order", np.arange(1)), dtype=int)
        if idx not in queue_order:
            return 0.0, 1.0, 1.0
        rank = int(np.where(queue_order == idx)[0][0])
        n_agents = max(len(queue_order) - 1, 1)
        leader_priority = clip01(1.0 - rank / n_agents)
        slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_state]
        forward_room = clip01(0.5 * (gap_balance + 1.0))
        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
        slot_alignment = clip01(1.0 - abs(slot_error))
        lateral_alignment = clip01(1.0 - abs(lateral_offset))
        approach_role = clip01(0.4 * leader_priority + 0.25 * slot_alignment + 0.2 * lateral_alignment + 0.15 * obstacle_clearance)
        local_release = clip01(0.26 * leader_priority + 0.24 * forward_room + 0.2 * slot_alignment + 0.15 * obstacle_clearance + 0.15 * lateral_alignment)
        approach_norm = clip01(self.approach_energy_tank / max(self.params.magnetic_approach_tank_capacity, EPS))
        transport_norm = clip01(self.transport_energy_tank / max(self.params.magnetic_tank_capacity, EPS))
        entry_progress = clip01(float(global_phase.get("entry_progress", 0.0)))
        approach_budget = clip01(approach_norm * approach_role * (1.0 - 0.55 * crowding) * (1.0 - 0.65 * entry_progress))
        transport_budget = clip01(transport_norm * queue_drive_scale * (0.35 + 0.65 * local_release))
        if self.method.use_leader_driven_entry_bonus:
            approach_budget = clip01(0.82 * propagated_entry_budget + 0.18 * approach_budget)
            transport_budget = clip01(0.35 * transport_budget)
        elif self.method.use_energy_global_quota_redistribution:
            approach_budget = clip01(0.7 * propagated_entry_budget + 0.3 * approach_budget)
            transport_budget = clip01(0.65 * transport_budget)
        elif not self.method.use_energy_xi_allocation:
            approach_budget = clip01(approach_norm * queue_drive_scale)
            transport_budget = clip01(transport_norm * queue_drive_scale)
        if self.method.use_entry_energy_propagation:
            approach_budget = clip01(max(approach_budget, propagated_entry_budget))
        budget = clip01(approach_budget + transport_budget)
        budget_effect = self._dual_stage_activation(budget)
        drive_gain = float(
            np.clip(
                self.params.magnetic_tank_drive_floor
                + (self.params.magnetic_tank_drive_ceiling - self.params.magnetic_tank_drive_floor) * budget_effect,
                self.params.magnetic_tank_drive_floor,
                self.params.magnetic_tank_drive_ceiling,
            )
        )
        curl_gain = float(
            np.clip(
                self.params.magnetic_tank_curl_floor
                + (self.params.magnetic_tank_curl_ceiling - self.params.magnetic_tank_curl_floor) * budget_effect,
                self.params.magnetic_tank_curl_floor,
                self.params.magnetic_tank_curl_ceiling,
            )
        )
        if not self.method.use_energy_curl_efficiency:
            curl_gain = 1.0
        return budget, drive_gain, curl_gain

    def _transport_tank_release_drive(
        self,
        idx: int,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
        queue_drive_scale: float,
        propagated_entry_budget: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        if not (self.method.use_energy_global_quota_redistribution or self.method.use_entry_energy_propagation) or global_phase is None:
            return np.zeros(2, dtype=float), 0.0
        transport_norm = clip01(self.transport_energy_tank / max(self.params.magnetic_tank_capacity, EPS))
        if transport_norm <= EPS:
            return np.zeros(2, dtype=float), 0.0

        local_axis, _local_perp = self._xi_polarity_frame(idx, global_phase, xi_state)
        if norm(local_axis) <= EPS:
            return np.zeros(2, dtype=float), 0.0

        queue_order = np.asarray(global_phase.get("queue_order", np.arange(1)), dtype=int)
        if idx not in queue_order:
            return np.zeros(2, dtype=float), 0.0
        rank = int(np.where(queue_order == idx)[0][0])
        n_agents = max(len(queue_order) - 1, 1)
        leader_priority = clip01(1.0 - rank / n_agents)

        slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_state]
        forward_room = clip01(0.5 * (gap_balance + 1.0))
        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
        slot_alignment = clip01(1.0 - abs(slot_error))
        lateral_alignment = clip01(1.0 - abs(lateral_offset))
        local_release = clip01(
            0.26 * leader_priority
            + 0.24 * forward_room
            + 0.2 * slot_alignment
            + 0.15 * obstacle_clearance
            + 0.15 * lateral_alignment
        )

        corridor_commitment = clip01(float(global_phase.get("corridor_commitment", 0.0)))
        release_need = clip01(float(global_phase.get("release_need", 0.0)))
        release_ready = clip01(float(global_phase.get("xi_feedback_release_ready", 0.0)))
        entry_progress = clip01(float(global_phase.get("entry_progress", 0.0)))
        traverse_progress = clip01(float(global_phase.get("traverse_progress", 0.0)))
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))

        release_gate = clip01(
            0.28 * corridor_commitment
            + 0.26 * release_need
            + 0.18 * release_ready
            + 0.14 * forward_free_norm
            + 0.14 * traverse_progress
        )
        quota_gate = clip01(
            0.5 * propagated_entry_budget
            + 0.5 * queue_drive_scale * (0.35 + 0.65 * local_release)
        )
        transport_effect = transport_norm
        front_loaded_weight = 0.7 + 0.3 * leader_priority
        if self.method.use_front_loaded_transport_release:
            front_span = max(1, int(np.ceil(0.5 * len(queue_order))))
            front_loaded_weight = float(np.clip(1.0 - 0.85 * rank / max(front_span, 1), 0.15, 1.0))
            release_gate = clip01(
                max(
                    release_gate,
                    0.42 * entry_progress
                    + 0.24 * corridor_commitment
                    + 0.16 * slot_alignment
                    + 0.18 * propagated_entry_budget,
                )
            )
            quota_gate = clip01(
                max(
                    quota_gate,
                    0.85 * propagated_entry_budget
                    + 0.45 * queue_drive_scale * (0.35 + 0.65 * local_release),
                )
            )
            transport_effect = np.sqrt(transport_norm)
        if self.method.use_spring_chain_reduced_model:
            front_span = max(1, int(np.ceil(0.5 * len(queue_order))))
            support_profile = float(np.clip(1.0 - rank / max(front_span, 1), 0.0, 1.0))
            release_gate = clip01(max(release_gate, 0.7 * self.last_spring_head_release + 0.15 * entry_progress))
            quota_gate = clip01(max(quota_gate, 0.65 * self.last_spring_rear_support + 0.25 * propagated_entry_budget))
            front_loaded_weight = max(front_loaded_weight, 0.3 + 0.7 * support_profile)
            transport_effect = max(transport_effect, np.sqrt(transport_norm))
        scalar = (
            self.params.u_max
            * self.params.magnetic_transport_release_gain
            * (1.25 + 0.75 * front_loaded_weight)
            * transport_effect
            * (0.45 + 0.55 * release_gate)
            * quota_gate
            * front_loaded_weight
            * (0.85 + 0.15 * forward_free_norm)
            * (1.0 - 0.15 * crowding)
        )
        if self.method.use_spring_chain_reduced_model:
            front_span = max(1, int(np.ceil(0.5 * len(queue_order))))
            support_profile = float(np.clip(1.0 - rank / max(front_span, 1), 0.0, 1.0))
            head_release_gain = 1.0 + 1.45 * self.last_spring_head_release if rank == 0 else 1.0
            rear_support_gain = 1.0 + 1.1 * self.last_spring_rear_support * support_profile
            scalar *= head_release_gain * rear_support_gain * (0.85 + 0.35 * self.spring_chain_storage_state)
        return scalar * local_axis, float(scalar)

    def _update_spring_chain_observables(
        self,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_states: np.ndarray,
    ) -> dict[str, float]:
        if global_phase is None or xi_states.size == 0:
            self.spring_chain_storage_state = 0.0
            self.last_spring_chain_storage = 0.0
            self.last_spring_head_release = 0.0
            self.last_spring_rear_support = 0.0
            self.last_spring_link_compression = 0.0
            return {
                "spring_chain_storage": 0.0,
                "spring_head_release": 0.0,
                "spring_rear_support": 0.0,
                "spring_link_compression": 0.0,
            }

        queue_order = np.asarray(global_phase.get("queue_order", np.arange(len(xi_states))), dtype=int)
        parallel_coords = np.asarray(global_phase.get("parallel_coords", np.zeros(len(xi_states))), dtype=float)
        if len(queue_order) <= 1:
            self.spring_chain_storage_state = 0.0
            self.last_spring_chain_storage = 0.0
            self.last_spring_head_release = 0.0
            self.last_spring_rear_support = 0.0
            self.last_spring_link_compression = 0.0
            return {
                "spring_chain_storage": 0.0,
                "spring_head_release": 0.0,
                "spring_rear_support": 0.0,
                "spring_link_compression": 0.0,
            }

        desired_gap = max(self.params.geo_queue_spacing, EPS)
        compressions: list[float] = []
        weighted_front_sum = 0.0
        weight_total = 0.0
        for link_rank in range(len(queue_order) - 1):
            front_idx = int(queue_order[link_rank])
            back_idx = int(queue_order[link_rank + 1])
            gap = float(parallel_coords[front_idx] - parallel_coords[back_idx])
            compression = clip01((desired_gap - gap) / desired_gap)
            back_xi = xi_states[back_idx]
            back_align = clip01(1.0 - 0.5 * abs(float(back_xi[0])) - 0.5 * abs(float(back_xi[1])))
            compression *= 0.75 + 0.25 * back_align
            weight = 1.0 - 0.75 * link_rank / max(len(queue_order) - 1, 1)
            compressions.append(compression)
            weighted_front_sum += weight * compression
            weight_total += weight
        link_compression = float(np.mean(compressions)) if compressions else 0.0
        front_compression = float(weighted_front_sum / max(weight_total, EPS))

        corridor_commitment = clip01(float(global_phase.get("corridor_commitment", 0.0)))
        release_need = clip01(float(global_phase.get("release_need", 0.0)))
        release_ready = clip01(float(global_phase.get("xi_feedback_release_ready", 0.0)))
        entry_progress = clip01(float(global_phase.get("entry_progress", 0.0)))
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        obstacle_pressure = clip01(float(global_phase.get("obstacle_pressure", 0.0)))
        teammate_pressure = clip01(float(global_phase.get("teammate_pressure", 0.0)))

        storage_target = clip01(
            0.42 * link_compression
            + 0.26 * front_compression
            + 0.18 * corridor_commitment
            + 0.14 * (1.0 - obstacle_pressure)
        )
        self.spring_chain_storage_state = clip01(
            self.spring_chain_storage_state + self.params.dt * 2.0 * (storage_target - self.spring_chain_storage_state)
        )

        head_idx = int(queue_order[0])
        head_xi = xi_states[head_idx]
        head_alignment = clip01(1.0 - 0.5 * abs(float(head_xi[0])) - 0.5 * abs(float(head_xi[1])))
        head_release = clip01(
            0.32 * release_need
            + 0.22 * release_ready
            + 0.16 * forward_free_norm
            + 0.15 * entry_progress
            + 0.15 * head_alignment
        )
        rear_support = clip01(
            self.spring_chain_storage_state
            * (0.6 + 0.4 * front_compression)
            * (0.65 + 0.35 * teammate_pressure)
            * (0.9 - 0.25 * obstacle_pressure)
        )

        self.last_spring_chain_storage = self.spring_chain_storage_state
        self.last_spring_head_release = head_release
        self.last_spring_rear_support = rear_support
        self.last_spring_link_compression = link_compression
        return {
            "spring_chain_storage": self.spring_chain_storage_state,
            "spring_head_release": head_release,
            "spring_rear_support": rear_support,
            "spring_link_compression": link_compression,
        }

    def _entry_energy_propagation_budgets(
        self,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_states: np.ndarray,
    ) -> np.ndarray:
        if not self.method.use_entry_energy_propagation or global_phase is None or xi_states.size == 0:
            return np.zeros(len(xi_states), dtype=float)
        queue_order = np.asarray(global_phase.get("queue_order", np.arange(len(xi_states))), dtype=int)
        parallel_coords = np.asarray(global_phase.get("parallel_coords", np.zeros(len(xi_states))), dtype=float)
        desired_gap = max(self.params.geo_queue_spacing, EPS)
        entry_progress = clip01(float(global_phase.get("entry_progress", 0.0)))
        corridor_commitment = clip01(float(global_phase.get("corridor_commitment", 0.0)))
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))
        approach_norm = clip01(self.approach_energy_tank / max(self.params.magnetic_approach_tank_capacity, EPS))
        if len(queue_order) == 0:
            return np.zeros(len(xi_states), dtype=float)

        budgets = np.zeros(len(xi_states), dtype=float)
        leader_idx = int(queue_order[0])
        leader_xi = xi_states[leader_idx]
        leader_slot = clip01(1.0 - abs(float(leader_xi[0])))
        leader_obs = clip01(0.5 * (float(leader_xi[3]) + 1.0))
        leader_crowd = clip01(float(leader_xi[4]))
        if self.method.use_leader_driven_entry_bonus:
            leader_bonus = clip01(
                approach_norm
                * (0.95 + 0.55 * corridor_commitment)
                * (0.7 + 0.6 * forward_free_norm)
                * (0.75 + 0.45 * leader_slot)
                * (0.8 + 0.35 * leader_obs)
                * (1.0 - 0.2 * leader_crowd)
                * (1.0 - 0.35 * entry_progress)
            )
        else:
            leader_bonus = clip01(
                approach_norm
                * (0.55 + 0.45 * corridor_commitment)
                * (0.45 + 0.55 * forward_free_norm)
                * (0.55 + 0.45 * leader_slot)
                * (0.7 + 0.3 * leader_obs)
                * (1.0 - 0.35 * leader_crowd)
                * (1.0 - 0.55 * entry_progress)
            )
        budgets[leader_idx] = leader_bonus

        for rank in range(1, len(queue_order)):
            idx = int(queue_order[rank])
            front_idx = int(queue_order[rank - 1])
            front_gap = float(parallel_coords[front_idx] - parallel_coords[idx])
            gap_match = clip01(1.0 - abs(front_gap - desired_gap) / max(1.5 * desired_gap, EPS))
            xi = xi_states[idx]
            slot_align = clip01(1.0 - abs(float(xi[0])))
            lateral_align = clip01(1.0 - abs(float(xi[1])))
            obstacle_clearance = clip01(0.5 * (float(xi[3]) + 1.0))
            crowding = clip01(float(xi[4]))
            carry = budgets[front_idx]
            if self.method.use_leader_driven_entry_bonus:
                budgets[idx] = clip01(
                    carry
                    * (0.82 + 0.22 * slot_align)
                    * (0.82 + 0.18 * gap_match)
                    * (0.78 + 0.18 * lateral_align)
                    * (0.78 + 0.18 * obstacle_clearance)
                    * (1.0 - 0.18 * crowding)
                )
            else:
                budgets[idx] = clip01(
                    carry
                    * (0.72 + 0.18 * slot_align)
                    * (0.65 + 0.25 * gap_match)
                    * (0.7 + 0.2 * lateral_align)
                    * (0.7 + 0.2 * obstacle_clearance)
                    * (1.0 - 0.28 * crowding)
                )
        return budgets

    def _global_quota_xi_redistribution_budgets(
        self,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_states: np.ndarray,
    ) -> np.ndarray:
        if not self.method.use_energy_global_quota_redistribution or global_phase is None or xi_states.size == 0:
            return np.zeros(len(xi_states), dtype=float)
        queue_order = np.asarray(global_phase.get("queue_order", np.arange(len(xi_states))), dtype=int)
        if len(queue_order) == 0:
            return np.zeros(len(xi_states), dtype=float)
        approach_norm = clip01(self.approach_energy_tank / max(self.params.magnetic_approach_tank_capacity, EPS))
        transport_norm = clip01(self.transport_energy_tank / max(self.params.magnetic_tank_capacity, EPS))
        scene_signal = self.last_energy_scene_signal if self.method.use_energy_scene_layer else 1.0
        corridor_commitment = clip01(float(global_phase.get("corridor_commitment", 0.0)))
        release_need = clip01(float(global_phase.get("release_need", 0.0)))
        entry_progress = clip01(float(global_phase.get("entry_progress", 0.0)))
        forward_free_norm = clip01(float(global_phase.get("forward_free_length", 0.0)) / max(2.0 * self.params.r0, EPS))

        total_quota = clip01(
            (0.65 * approach_norm * (1.0 - 0.55 * entry_progress) + 0.35 * transport_norm)
            * (0.55 + 0.45 * scene_signal)
            * (0.55 + 0.45 * max(corridor_commitment, release_need))
            * (0.4 + 0.6 * forward_free_norm)
        )
        requests = np.zeros(len(xi_states), dtype=float)
        recovered = 0.0
        for rank, idx in enumerate(queue_order):
            idx = int(idx)
            n_agents = max(len(queue_order) - 1, 1)
            leader_priority = clip01(1.0 - rank / n_agents)
            slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_states[idx]]
            forward_room = clip01(0.5 * (gap_balance + 1.0))
            obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
            slot_align = clip01(1.0 - abs(slot_error))
            lateral_align = clip01(1.0 - abs(lateral_offset))
            request = clip01(
                0.38 * leader_priority
                + 0.2 * forward_room
                + 0.17 * slot_align
                + 0.12 * lateral_align
                + 0.13 * obstacle_clearance
            )
            occ_force, drive_scale = self._soft_queue_occupancy_force(idx, global_phase, xi_states[idx])
            del occ_force
            requests[idx] = request * drive_scale
            recovered += request * (1.0 - drive_scale)
        if float(np.sum(requests)) <= EPS:
            return np.zeros(len(xi_states), dtype=float)
        # redistribute recovered budget to the front half of the queue
        front_count = max(1, int(np.ceil(0.5 * len(queue_order))))
        for rank, idx in enumerate(queue_order[:front_count]):
            idx = int(idx)
            bonus = recovered * (1.0 - rank / max(front_count, 1)) / max(front_count, 1)
            requests[idx] += bonus
        weights = requests / max(float(np.sum(requests)), EPS)
        return np.clip(total_quota * weights * len(queue_order), 0.0, 1.0)

    def _dual_stage_activation(self, budget: float) -> float:
        x = clip01(float(budget))
        mode = self.method.dual_stage_activation_mode
        if not self.method.use_energy_activation_trigger:
            return x

        if mode == "linear":
            return np.sqrt(x)

        slope = max(self.params.magnetic_activation_slope, EPS)
        threshold = clip01(self.params.magnetic_activation_threshold)

        if mode == "sigmoid":
            lo = 1.0 / (1.0 + np.exp(slope * threshold))
            hi = 1.0 / (1.0 + np.exp(-slope * (1.0 - threshold)))
            val = 1.0 / (1.0 + np.exp(-slope * (x - threshold)))
            return clip01((val - lo) / max(hi - lo, EPS))

        if mode == "tanh":
            lo = np.tanh(-slope * threshold)
            hi = np.tanh(slope * (1.0 - threshold))
            val = np.tanh(slope * (x - threshold))
            return clip01((val - lo) / max(hi - lo, EPS))

        if mode == "window":
            low = clip01(self.params.magnetic_activation_window_low)
            high = max(low + 0.05, clip01(self.params.magnetic_activation_window_high))
            left = 1.0 / (1.0 + np.exp(-slope * (x - low)))
            right = 1.0 / (1.0 + np.exp(-slope * (high - x)))
            mid = 0.5 * (low + high)
            peak = (1.0 / (1.0 + np.exp(-slope * (mid - low)))) * (1.0 / (1.0 + np.exp(-slope * (high - mid))))
            return clip01((left * right) / max(peak, EPS))

        return np.sqrt(x)

    def _unified_state_observables(
        self,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_axes: np.ndarray,
        xi_states: np.ndarray,
        transport_energy: float,
        approach_energy: float,
        scenario: Scenario | None,
    ) -> dict[str, float]:
        if global_phase is None:
            return {"r": np.nan, "w": np.nan, "q": np.nan, "a": np.nan, "e": np.nan}

        if scenario is not None and scenario.corridor_exit_x is not None and self.params_obstacles:
            corridor_start_x = min(float(obstacle.center[0] - obstacle.radius) for obstacle in self.params_obstacles)
        else:
            corridor_start_x = float(np.asarray(global_phase.get("center", np.zeros(2)))[0])
        front_x = float(np.max(np.asarray(global_phase.get("center", np.zeros(2)))[0] + np.asarray(global_phase.get("parallel_coords", np.zeros(1)))))
        axis = unit(np.asarray(global_phase.get("axis", np.array([1.0, 0.0])), dtype=float))
        if norm(axis) <= EPS:
            axis = np.array([1.0, 0.0], dtype=float)

        r = front_x - corridor_start_x
        w = float(global_phase.get("lateral_width", np.nan))

        slot_order = clip01(float(global_phase.get("xi_feedback_slot_order_readiness", 0.0)))
        lateral_order = clip01(float(global_phase.get("xi_feedback_lateral_order_readiness", 0.0)))
        teammate_pressure = clip01(float(global_phase.get("teammate_pressure", 0.0)))
        longitudinal_span = max(float(global_phase.get("longitudinal_span", 0.0)), EPS)
        queue_shape = clip01(1.0 - w / max(longitudinal_span + 0.45 * self.params.r0, EPS)) if np.isfinite(w) else 0.0
        q = clip01(0.35 * slot_order + 0.25 * lateral_order + 0.2 * queue_shape + 0.2 * (1.0 - teammate_pressure))

        alignments = []
        for vec in xi_axes:
            vec = np.asarray(vec, dtype=float)
            if np.any(np.isnan(vec)) or norm(vec) <= EPS:
                continue
            alignments.append(abs(float(unit(vec) @ axis)))
        a = float(np.mean(alignments)) if alignments else 0.0

        e = clip01(float(transport_energy) + 0.45 * float(approach_energy))
        return {"r": float(r), "w": float(w), "q": float(q), "a": float(a), "e": float(e)}

    def _unified_state_entry_push(self, unified_state: dict[str, float], axis: np.ndarray) -> np.ndarray:
        r = float(unified_state["r"])
        if not np.isfinite(r) or r >= 0.0:
            return np.zeros(2, dtype=float)
        axis = unit(axis)
        if norm(axis) <= EPS:
            return np.zeros(2, dtype=float)
        e = clip01(float(unified_state["e"]))
        q = clip01(float(unified_state["q"]))
        a = clip01(float(unified_state["a"]))
        w = max(float(unified_state["w"]), 0.0)
        slope = max(self.params.unified_entry_slope, EPS)
        e_gate = 1.0 / (1.0 + np.exp(-slope * (e - self.params.unified_entry_energy_threshold)))
        w_gate = 1.0 / (1.0 + np.exp(slope * (w - self.params.unified_entry_width_ref)))
        scalar = self.params.unified_entry_push_gain * self.params.u_max * e_gate * q * a * w_gate
        return scalar * axis

    def _structured_energy_agent_budget(
        self,
        idx: int,
        global_phase: dict[str, np.ndarray | float] | None,
        xi_state: np.ndarray,
        queue_drive_scale: float,
    ) -> tuple[float, float, float]:
        if not self.method.use_structured_energy_tank or global_phase is None:
            return 0.0, 1.0, 1.0
        tank_norm = clip01(self.transport_energy_tank / max(self.params.magnetic_tank_capacity, EPS))
        queue_order = np.asarray(global_phase.get("queue_order", np.arange(1)), dtype=int)
        if idx not in queue_order:
            return 0.0, 1.0, 1.0
        rank = int(np.where(queue_order == idx)[0][0])
        n_agents = max(len(queue_order) - 1, 1)
        leader_priority = clip01(1.0 - rank / n_agents)
        slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_state]
        forward_room = clip01(0.5 * (gap_balance + 1.0))
        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
        slot_alignment = clip01(1.0 - abs(slot_error))
        lateral_alignment = clip01(1.0 - abs(lateral_offset))
        local_release = clip01(
            0.26 * leader_priority
            + 0.24 * forward_room
            + 0.2 * slot_alignment
            + 0.15 * obstacle_clearance
            + 0.15 * lateral_alignment
        )
        budget = clip01(tank_norm * queue_drive_scale * (0.35 + 0.65 * local_release))
        budget_effect = np.sqrt(budget)
        drive_gain = float(
            np.clip(
                self.params.magnetic_tank_drive_floor
                + (self.params.magnetic_tank_drive_ceiling - self.params.magnetic_tank_drive_floor) * budget_effect,
                self.params.magnetic_tank_drive_floor,
                self.params.magnetic_tank_drive_ceiling,
            )
        )
        curl_gain = float(
            np.clip(
                self.params.magnetic_tank_curl_floor
                + (self.params.magnetic_tank_curl_ceiling - self.params.magnetic_tank_curl_floor) * budget_effect,
                self.params.magnetic_tank_curl_floor,
                self.params.magnetic_tank_curl_ceiling,
            )
        )
        return budget, drive_gain, curl_gain

    def _corridor_linear_constraints(
        self,
        idx: int,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        global_phase: dict[str, np.ndarray | float] | None,
    ) -> list[tuple[np.ndarray, float]]:
        constraints: list[tuple[np.ndarray, float]] = []
        tau = max(0.4, 8.0 * self.params.dt)
        position = positions[idx]
        neighbors = self._neighbor_sets(positions)
        stage_strength = self._corridor_low_level_stage_strength(global_phase)

        for j in neighbors[idx]:
            rel = position - positions[j]
            dist = norm(rel)
            if dist <= EPS:
                continue
            if dist > self.params.rc + self.params.u_max * tau:
                continue
            normal = unit(rel)
            clearance = dist - self.params.d_agent_safe
            limit = float((-normal) @ preferred_velocities[j]) + max(clearance, -0.25 * self.params.d_agent_safe) / tau
            limit += 0.1 * (1.0 - stage_strength)
            constraints.append((-normal, limit))

        for obstacle in self.params_obstacles:
            delta = position - obstacle.center
            center_dist = norm(delta)
            if center_dist <= EPS:
                continue
            effective_radius = obstacle.radius + self.params.d_obs_safe
            clearance = center_dist - effective_radius
            if clearance > self.params.d_s + self.params.u_max * tau:
                continue
            normal = unit(delta)
            limit = max(clearance, -0.25 * self.params.d_obs_safe) / tau
            constraints.append((-normal, limit))

        return constraints

    def _project_velocity_sequence(
        self,
        preferred: np.ndarray,
        constraints: list[tuple[np.ndarray, float]],
        radius: float,
    ) -> np.ndarray:
        velocity = preferred.astype(float).copy()
        speed = norm(velocity)
        if speed > radius:
            velocity = radius * velocity / (speed + EPS)
        for normal, limit in constraints:
            violation = float(normal @ velocity) - limit
            if violation <= 1e-9:
                continue
            velocity = velocity - violation * normal / max(float(normal @ normal), EPS)
            speed = norm(velocity)
            if speed > radius:
                velocity = radius * velocity / (speed + EPS)
        return velocity

    def _corridor_projection_filter(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        global_phase: dict[str, np.ndarray | float],
    ) -> np.ndarray:
        filtered = preferred_velocities.copy()
        for i in range(len(positions)):
            constraints = self._corridor_linear_constraints(i, positions, preferred_velocities, global_phase)
            filtered[i] = self._project_velocity_sequence(preferred_velocities[i], constraints, self.params.u_max)
        return filtered

    def _corridor_qp_filter(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        global_phase: dict[str, np.ndarray | float],
    ) -> np.ndarray:
        filtered = preferred_velocities.copy()
        for i in range(len(positions)):
            constraints = self._corridor_linear_constraints(i, positions, preferred_velocities, global_phase)
            filtered[i] = self._solve_velocity_qp(preferred_velocities[i], constraints, self.params.u_max)
        return filtered

    def _tensor_environment_summary(self, position: np.ndarray, predicted_target: np.ndarray) -> tuple[np.ndarray, float, float]:
        nav_axis = unit(predicted_target - position)
        if norm(nav_axis) <= EPS:
            nav_axis = np.array([1.0, 0.0], dtype=float)
        blocked = np.zeros((2, 2), dtype=float)
        total_pressure = 0.0
        for obstacle in self.params_obstacles:
            d, normal = self._obstacle_distance(position, obstacle)
            xi = clip01((self.params.d_s - d) / max(self.params.d_s - self.params.d_obs_safe, EPS))
            if xi <= EPS:
                continue
            blocked += xi * np.outer(normal, normal)
            total_pressure += xi
        if total_pressure <= EPS:
            return nav_axis, 0.0, 0.0
        evals, evecs = np.linalg.eigh(blocked)
        axis = evecs[:, int(np.argmin(evals))]
        if float(axis @ nav_axis) < 0.0:
            axis = -axis
        pressure = clip01(total_pressure / max(len(self.params_obstacles), 1))
        anisotropy = clip01((float(np.max(evals)) - float(np.min(evals))) / (float(np.sum(np.abs(evals))) + EPS))
        blended = unit((1.0 - 0.45 * pressure) * nav_axis + (0.35 + 0.65 * anisotropy) * axis)
        return (blended if norm(blended) > EPS else axis, pressure, anisotropy)

    def _tensor_environment_matrix(self, position: np.ndarray, predicted_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        nav_axis = unit(predicted_target - position)
        if norm(nav_axis) <= EPS:
            nav_axis = np.array([1.0, 0.0], dtype=float)
        blocked = np.zeros((2, 2), dtype=float)
        total_pressure = 0.0
        for obstacle in self.params_obstacles:
            d, normal = self._obstacle_distance(position, obstacle)
            xi = clip01((self.params.d_s - d) / max(self.params.d_s - self.params.d_obs_safe, EPS))
            if xi <= EPS:
                continue
            blocked += xi * np.outer(normal, normal)
            total_pressure += xi
        if total_pressure <= EPS:
            return np.outer(nav_axis, nav_axis), nav_axis, 0.0, 0.0
        evals, evecs = np.linalg.eigh(blocked)
        free_axis = evecs[:, int(np.argmin(evals))]
        if float(free_axis @ nav_axis) < 0.0:
            free_axis = -free_axis
        pressure = clip01(total_pressure / max(len(self.params_obstacles), 1))
        anisotropy = clip01((float(np.max(evals)) - float(np.min(evals))) / (float(np.sum(np.abs(evals))) + EPS))
        free_tensor = np.outer(free_axis, free_axis)
        nav_tensor = np.outer(nav_axis, nav_axis)
        env_matrix = (self.params.tensor_env_keep * free_tensor + (1.0 - self.params.tensor_env_keep) * nav_tensor)
        trace = float(np.trace(env_matrix))
        if trace > EPS:
            env_matrix = env_matrix / trace
        return env_matrix, free_axis, pressure, anisotropy

    def _tensor_formation_summary(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        predicted_target: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        fallback = unit(predicted_target - np.mean(positions, axis=0))
        if norm(fallback) <= EPS:
            fallback = np.array([1.0, 0.0], dtype=float)
        if not neighbors:
            return fallback, 0.0, 0.0
        cov = np.zeros((2, 2), dtype=float)
        for j in neighbors:
            rel = positions[j] - positions[idx]
            scale = 1.0 / max(norm(rel), 0.5 * self.params.r0, EPS)
            cov += scale * np.outer(rel, rel)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        if float(axis @ fallback) < 0.0:
            axis = -axis
        anisotropy = clip01((float(np.max(evals)) - float(np.min(evals))) / (float(np.sum(np.abs(evals))) + EPS))
        spread = clip01(float(np.trace(cov)) / max(len(neighbors) * self.params.r0, EPS))
        return axis, anisotropy, spread

    def _tensor_formation_matrix(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        predicted_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        fallback = unit(predicted_target - np.mean(positions, axis=0))
        if norm(fallback) <= EPS:
            fallback = np.array([1.0, 0.0], dtype=float)
        if not neighbors:
            return np.outer(fallback, fallback), fallback, 0.0, 0.0
        cov = np.zeros((2, 2), dtype=float)
        for j in neighbors:
            rel = positions[j] - positions[idx]
            scale = 1.0 / max(norm(rel), 0.5 * self.params.r0, EPS)
            cov += scale * np.outer(rel, rel)
        trace = float(np.trace(cov))
        if trace > EPS:
            cov = cov / trace
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        if float(axis @ fallback) < 0.0:
            axis = -axis
        anisotropy = clip01((float(np.max(evals)) - float(np.min(evals))) / (float(np.sum(np.abs(evals))) + EPS))
        spread = clip01(float(np.trace(cov)) / max(len(neighbors) * self.params.r0, EPS))
        return cov, axis, anisotropy, spread

    def _tensor_energy_polar_summary(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        position: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float, float, dict[str, np.ndarray | float]]:
        env_matrix, env_axis, env_pressure, env_anisotropy = self._tensor_environment_matrix(position, predicted_target)
        form_matrix, form_axis, form_anisotropy, _form_spread = self._tensor_formation_matrix(idx, positions, neighbors, predicted_target)
        goal_axis = unit(predicted_target - position)
        if norm(goal_axis) <= EPS:
            goal_axis = env_axis
        queue_mode = max(env_pressure, clip01(norm(target.velocity) / max(self.params.geo_speed_trigger, EPS)))
        polar_tensor = (
            self.params.tensor_polar_env_weight * env_pressure * np.outer(env_axis, env_axis)
            + self.params.tensor_polar_form_weight * form_anisotropy * np.outer(form_axis, form_axis)
            + self.params.tensor_polar_goal_weight * queue_mode * np.outer(goal_axis, goal_axis)
        )
        polar_trace = float(np.trace(polar_tensor))
        if polar_trace > EPS:
            polar_tensor = polar_tensor / polar_trace
        combined = env_matrix + form_matrix + polar_tensor
        trace = float(np.trace(combined))
        if trace > EPS:
            combined = combined / trace
        evals, evecs = np.linalg.eigh(combined)
        axis = evecs[:, int(np.argmax(evals))]
        if float(axis @ goal_axis) < 0.0:
            axis = -axis
        anisotropy = clip01((float(np.max(evals)) - float(np.min(evals))) / (float(np.sum(np.abs(evals))) + EPS))
        parallel_gain = 1.0 + self.params.tensor_parallel_gain * max(queue_mode, anisotropy)
        perp_gain = max(self.params.tensor_perp_floor, 1.0 - (1.0 - self.params.tensor_perp_floor) * max(env_pressure, 0.6 * anisotropy))
        modulation = self._directional_matrix(axis, parallel_gain, perp_gain)
        enc_scale = max(0.12, 1.0 - 0.75 * env_pressure - 0.45 * queue_mode)
        angle_scale = max(0.2, 1.0 - 0.55 * env_pressure - 0.25 * queue_mode)
        details = self._empty_understanding_details()
        details.update(
            {
                "env_axis": env_axis,
                "form_axis": form_axis,
                "tensor_axis": axis,
                "env_pressure": env_pressure,
                "env_anisotropy": env_anisotropy,
                "form_anisotropy": form_anisotropy,
            }
        )
        return modulation, enc_scale, angle_scale, env_pressure, form_anisotropy, details

    def _continuous_polar_mode_summary(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        position: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float, float, dict[str, np.ndarray | float]]:
        env_matrix, env_axis, env_pressure, env_anisotropy = self._tensor_environment_matrix(position, predicted_target)
        form_matrix, form_axis, form_anisotropy, _form_spread = self._tensor_formation_matrix(idx, positions, neighbors, predicted_target)
        goal_axis = unit(predicted_target - position)
        if norm(goal_axis) <= EPS:
            goal_axis = env_axis
        center = (1.0 - self.params.beta_lead) * target.position + self.params.beta_lead * predicted_target
        radial_axis = unit(position - center)
        if norm(radial_axis) <= EPS:
            radial_axis = form_axis
        tangential_axis = rotate90(radial_axis)
        speed_norm = clip01(norm(target.velocity) / max(self.params.geo_speed_trigger, EPS))
        queue_mode = max(env_pressure, speed_norm)
        open_space = clip01(1.0 - env_pressure)
        recovery_need = clip01(form_anisotropy * open_space * (1.0 - speed_norm))

        encircle_logit = 1.4 * open_space + 0.8 * (1.0 - speed_norm) + 0.4 * (1.0 - form_anisotropy)
        corridor_logit = 1.6 * env_pressure + 0.9 * env_anisotropy + 0.2 * (1.0 - form_anisotropy)
        queue_logit = 1.5 * speed_norm + 0.6 * env_anisotropy + 0.5 * form_anisotropy
        recover_logit = 1.0 * recovery_need + 0.5 * open_space + 0.4 * form_anisotropy
        mode_weights = self._softmax(
            np.array([encircle_logit, corridor_logit, queue_logit, recover_logit], dtype=float),
            temperature=self.params.polar_mode_temperature,
        )

        encircle_tensor = 0.65 * np.outer(tangential_axis, tangential_axis) + 0.35 * np.outer(radial_axis, radial_axis)
        corridor_tensor = np.outer(env_axis, env_axis)
        queue_tensor = np.outer(goal_axis, goal_axis)
        recover_axis = unit(0.6 * tangential_axis + 0.4 * rotate90(goal_axis))
        if norm(recover_axis) <= EPS:
            recover_axis = tangential_axis
        recover_tensor = 0.6 * np.outer(recover_axis, recover_axis) + 0.4 * np.outer(radial_axis, radial_axis)
        polar_tensor = (
            mode_weights[0] * encircle_tensor
            + mode_weights[1] * corridor_tensor
            + mode_weights[2] * queue_tensor
            + mode_weights[3] * recover_tensor
        )
        polar_trace = float(np.trace(polar_tensor))
        if polar_trace > EPS:
            polar_tensor = polar_tensor / polar_trace

        combined = (
            env_matrix
            + self.params.tensor_polar_form_weight * form_matrix
            + self.params.tensor_polar_goal_weight * polar_tensor
        )
        trace = float(np.trace(combined))
        if trace > EPS:
            combined = combined / trace
        evals, evecs = np.linalg.eigh(combined)
        axis = evecs[:, int(np.argmax(evals))]
        if float(axis @ goal_axis) < 0.0:
            axis = -axis
        anisotropy = clip01((float(np.max(evals)) - float(np.min(evals))) / (float(np.sum(np.abs(evals))) + EPS))
        corridor_weight = float(mode_weights[1])
        queue_weight = float(mode_weights[2])
        parallel_gain = 1.0 + self.params.tensor_parallel_gain * max(corridor_weight, queue_weight, anisotropy)
        perp_gain = max(self.params.tensor_perp_floor, 1.0 - (1.0 - self.params.tensor_perp_floor) * max(corridor_weight, 0.6 * queue_weight))
        modulation = self._directional_matrix(axis, parallel_gain, perp_gain)
        enc_scale = max(0.12, 0.9 * mode_weights[0] + 0.65 * mode_weights[3] + 0.2)
        angle_scale = max(0.18, 0.85 * mode_weights[0] + 0.55 * mode_weights[3] + 0.15)
        details = self._empty_understanding_details()
        details.update(
            {
                "env_axis": env_axis,
                "form_axis": form_axis,
                "tensor_axis": axis,
                "env_pressure": env_pressure,
                "env_anisotropy": env_anisotropy,
                "form_anisotropy": form_anisotropy,
                "mode_logits": np.array([encircle_logit, corridor_logit, queue_logit, recover_logit], dtype=float),
                "mode_weights": mode_weights.astype(float),
            }
        )
        return modulation, enc_scale, angle_scale, env_pressure, form_anisotropy, details

    def _global_phase_guided_tensor_summary(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        position: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
        global_phase: dict[str, np.ndarray | float],
    ) -> tuple[np.ndarray, float, float, float, float, dict[str, np.ndarray | float]]:
        env_matrix, env_axis, env_pressure, env_anisotropy = self._tensor_environment_matrix(position, predicted_target)
        form_matrix, form_axis, form_anisotropy, _form_spread = self._tensor_formation_matrix(idx, positions, neighbors, predicted_target)
        goal_axis = unit(predicted_target - position)
        if norm(goal_axis) <= EPS:
            goal_axis = env_axis
        center = (1.0 - self.params.beta_lead) * target.position + self.params.beta_lead * predicted_target
        radial_axis = unit(position - center)
        if norm(radial_axis) <= EPS:
            radial_axis = form_axis
        tangential_axis = rotate90(radial_axis)
        mode_weights = np.asarray(global_phase["mode_weights"], dtype=float)
        if mode_weights.shape != (4,):
            mode_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

        encircle_tensor = 0.65 * np.outer(tangential_axis, tangential_axis) + 0.35 * np.outer(radial_axis, radial_axis)
        corridor_tensor = np.outer(env_axis, env_axis)
        queue_tensor = np.outer(goal_axis, goal_axis)
        recover_axis = unit(0.6 * tangential_axis + 0.4 * rotate90(goal_axis))
        if norm(recover_axis) <= EPS:
            recover_axis = tangential_axis
        recover_tensor = 0.6 * np.outer(recover_axis, recover_axis) + 0.4 * np.outer(radial_axis, radial_axis)
        polar_tensor = (
            mode_weights[0] * encircle_tensor
            + mode_weights[1] * corridor_tensor
            + mode_weights[2] * queue_tensor
            + mode_weights[3] * recover_tensor
        )
        polar_trace = float(np.trace(polar_tensor))
        if polar_trace > EPS:
            polar_tensor = polar_tensor / polar_trace

        combined = (
            env_matrix
            + self.params.tensor_polar_form_weight * form_matrix
            + self.params.tensor_polar_goal_weight * polar_tensor
        )
        trace = float(np.trace(combined))
        if trace > EPS:
            combined = combined / trace
        evals, evecs = np.linalg.eigh(combined)
        axis = evecs[:, int(np.argmax(evals))]
        if float(axis @ goal_axis) < 0.0:
            axis = -axis
        anisotropy = clip01((float(np.max(evals)) - float(np.min(evals))) / (float(np.sum(np.abs(evals))) + EPS))
        corridor_weight = float(mode_weights[1])
        queue_weight = float(mode_weights[2])
        parallel_gain = 1.0 + self.params.tensor_parallel_gain * max(corridor_weight, queue_weight, anisotropy)
        perp_gain = max(self.params.tensor_perp_floor, 1.0 - (1.0 - self.params.tensor_perp_floor) * max(corridor_weight, 0.6 * queue_weight))
        modulation = self._directional_matrix(axis, parallel_gain, perp_gain)
        enc_scale = max(0.12, 0.9 * mode_weights[0] + 0.65 * mode_weights[3] + 0.2)
        angle_scale = max(0.18, 0.85 * mode_weights[0] + 0.55 * mode_weights[3] + 0.15)
        details = self._empty_understanding_details()
        details.update(
            {
                "env_axis": env_axis,
                "form_axis": form_axis,
                "tensor_axis": axis,
                "env_pressure": env_pressure,
                "env_anisotropy": env_anisotropy,
                "form_anisotropy": form_anisotropy,
                "mode_logits": np.asarray(global_phase["mode_logits"], dtype=float),
                "mode_weights": mode_weights.astype(float),
            }
        )
        return modulation, enc_scale, angle_scale, env_pressure, form_anisotropy, details

    def _tensor_modulation_summary(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        position: np.ndarray,
        target: Target,
        predicted_target: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float, float, dict[str, np.ndarray | float]]:
        env_axis, env_pressure, env_anisotropy = self._tensor_environment_summary(position, predicted_target)
        form_axis, form_anisotropy, _ = self._tensor_formation_summary(idx, positions, neighbors, predicted_target)
        alignment = abs(float(env_axis @ form_axis))
        carry = form_anisotropy * alignment
        desired_axis = unit((1.0 - carry) * env_axis + carry * form_axis)
        if norm(desired_axis) <= EPS:
            desired_axis = env_axis
        speed_norm = clip01(norm(target.velocity) / max(self.params.geo_speed_trigger, EPS))
        parallel_gain = 1.0 + self.params.tensor_parallel_gain * max(env_pressure, speed_norm)
        perp_gain = max(self.params.tensor_perp_floor, 1.0 - (1.0 - self.params.tensor_perp_floor) * env_pressure - 0.25 * (1.0 - form_anisotropy))
        tensor = self._directional_matrix(desired_axis, parallel_gain, perp_gain)
        enc_scale = max(0.18, 1.0 - 0.75 * env_pressure - 0.35 * (1.0 - form_anisotropy))
        angle_scale = max(0.25, 1.0 - 0.55 * env_pressure)
        details = self._empty_understanding_details()
        details.update(
            {
                "env_axis": env_axis,
                "form_axis": form_axis,
                "tensor_axis": desired_axis,
                "env_pressure": env_pressure,
                "env_anisotropy": env_anisotropy,
                "form_anisotropy": form_anisotropy,
            }
        )
        return tensor, enc_scale, angle_scale, env_pressure, form_anisotropy, details

    def _projection_priority_force(
        self,
        nav: np.ndarray,
        rep: np.ndarray,
        curl: np.ndarray,
        safe: np.ndarray,
        topo: np.ndarray,
        enc: np.ndarray,
        ang: np.ndarray,
        directional: np.ndarray,
        queue_guard: np.ndarray,
    ) -> np.ndarray:
        high = rep + safe
        mid = self.params.projection_mid_gain * (directional + queue_guard + topo)
        mid_proj = self._nullspace_project(mid, high)
        low = self.params.projection_low_gain * (nav + enc + ang + curl)
        low_proj = self._nullspace_project(low, high + mid_proj)
        return high + mid_proj + low_proj

    def _fusion_force(
        self,
        idx: int,
        positions: np.ndarray,
        neighbors: list[int],
        target: Target,
        predicted_target: np.ndarray,
        omega_i: float,
        rho_i: float,
        phi_i: float,
        nav: np.ndarray,
        rep: np.ndarray,
        safe: np.ndarray,
        topo: np.ndarray,
        enc: np.ndarray,
        ang: np.ndarray,
        curl: np.ndarray,
        directional: np.ndarray,
        queue_guard: np.ndarray,
        corridor_activation: float,
        queue_activation: float,
        lateral_width: float,
        longitudinal_span: float,
        lateral_offset: float,
    ) -> np.ndarray:
        queue_mode = max(corridor_activation, queue_activation)
        geometry_phase = clip01(1.0 - queue_mode)
        speed_norm = clip01(norm(target.velocity) / max(self.params.geo_speed_trigger, EPS))
        width_norm = clip01(lateral_width / max(len(positions) * 0.5 * self.params.r0, EPS))
        span_norm = clip01(longitudinal_span / max(len(positions) * self.params.geo_queue_spacing, EPS))
        lateral_offset_norm = clip01(abs(lateral_offset) / max(self.params.r0, EPS))
        spacing_risk = self._local_spacing_risk(idx, positions, neighbors)

        energy_base = self._energy_flow_force(nav, rep, safe, topo, enc, ang, omega_i, rho_i, queue_mode)
        safe_anchor = self._blend_force(rep, 1.15) + self._blend_force(safe, 1.2)
        module_vectors = np.stack(
            [
                energy_base,
                self._blend_force(directional, 1.0),
                self._blend_force(queue_guard, 1.0),
                self._blend_force(nav, 1.0),
                self._blend_force(topo, 1.0),
                self._blend_force(enc, geometry_phase),
                self._blend_force(ang, geometry_phase),
                self._blend_force(curl, 0.9),
                safe_anchor,
            ],
            axis=0,
        )
        features = np.array(
            [
                1.0,
                omega_i,
                clip01(1.0 - rho_i),
                corridor_activation,
                queue_activation,
                speed_norm,
                width_norm,
                span_norm,
                spacing_risk,
                clip01(phi_i),
                lateral_offset_norm,
                geometry_phase,
            ],
            dtype=float,
        )
        module_keys = np.array(
            [
                [1.0, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.4, 0.3, 0.1, 0.2],
                [0.3, 0.2, 0.4, 1.2, 0.9, 0.4, 1.1, 0.5, 0.2, 0.2, 0.9, 0.0],
                [0.2, 0.1, 0.3, 0.7, 1.0, 0.5, 0.3, 1.0, 1.2, 0.1, 0.2, 0.0],
                [0.4, 0.0, 0.0, -0.3, 0.6, 1.1, 0.0, 0.0, -0.4, 0.0, 0.0, 0.4],
                [0.5, 0.2, 0.8, 0.5, 0.4, 0.2, 0.3, 0.7, 0.1, 0.1, 0.0, 0.3],
                [0.3, -0.3, -0.4, -0.5, -0.6, 0.1, 0.0, 0.2, -0.5, -0.1, 0.0, 1.3],
                [0.2, -0.2, -0.3, -0.5, -0.5, 0.0, 0.1, 0.1, -0.4, -0.1, 0.0, 1.2],
                [0.3, 0.8, 0.0, 0.8, 0.3, 0.1, 0.2, 0.0, 0.1, 1.1, 0.2, -0.2],
                [0.5, 1.1, 0.4, 0.3, 0.2, 0.0, 0.2, 0.3, 1.4, 0.8, 0.1, -0.2],
            ],
            dtype=float,
        )
        head_queries = np.array(
            [
                [0.3, 1.2, 0.3, 0.5, 0.1, 0.0, 0.2, 0.1, 1.3, 0.8, 0.1, -0.2],
                [0.2, 0.5, 0.7, 1.3, 0.5, 0.2, 1.0, 0.4, 0.3, 0.2, 0.8, 0.0],
                [0.2, 0.1, 0.2, 0.3, 1.2, 1.1, 0.3, 0.8, 0.3, 0.0, 0.2, 0.0],
                [0.3, -0.2, -0.1, -0.4, -0.5, 0.1, 0.0, 0.1, -0.4, -0.1, 0.0, 1.4],
            ],
            dtype=float,
        )
        head_biases = np.array(
            [
                [0.4, -0.2, -0.2, -0.4, -0.3, -0.8, -0.8, 0.1, 0.6],
                [0.2, 0.6, 0.3, -0.2, 0.2, -0.6, -0.7, 0.0, 0.1],
                [0.3, 0.2, 0.5, 0.2, 0.2, -0.5, -0.6, -0.2, -0.1],
                [0.2, -0.5, -0.6, 0.0, 0.2, 0.8, 0.7, -0.1, -0.2],
            ],
            dtype=float,
        )
        head_gate_weights = np.array(
            [
                [0.0, 1.0, 0.2, 0.3, 0.0, 0.0, 0.3, 0.2, 1.1, 0.7, 0.1, -0.2],
                [0.0, 0.3, 0.5, 1.0, 0.3, 0.1, 0.9, 0.3, 0.2, 0.2, 0.6, 0.0],
                [0.0, 0.1, 0.2, 0.2, 1.0, 0.9, 0.2, 0.7, 0.2, 0.0, 0.1, 0.0],
                [0.0, -0.2, -0.2, -0.5, -0.6, 0.1, 0.0, 0.1, -0.4, -0.1, 0.0, 1.1],
            ],
            dtype=float,
        )

        gate_logits = head_gate_weights @ features
        head_gates = self._softmax(gate_logits)
        fused = np.zeros(2)
        for head_idx, gate in enumerate(head_gates):
            query = head_queries[head_idx] * features
            module_logits = head_biases[head_idx] + module_keys @ query
            module_attention = self._softmax(module_logits)
            fused += gate * (module_attention @ module_vectors)
        return fused

    def _local_psi(self, idx: int, positions: np.ndarray, target: Target, neighbors: list[int], n_agents: int) -> tuple[float, float, float]:
        if not self.method.use_psi:
            return 1.0, 1.0, 1.0
        n_i = unit(positions[idx] - target.position)
        acc = n_i.copy()
        for j in neighbors:
            acc += unit(positions[j] - target.position)
        hat_psi = clip01(norm(acc) / max(len(neighbors) + 1, 1))
        c_i = clip01(len(neighbors) / max(self.params.n_min, 1))
        tilde_psi = c_i * hat_psi + (1.0 - c_i)
        d_val = self.params.delta + (1.0 - self.params.delta) * (tilde_psi ** self.params.q_psi)
        return hat_psi, tilde_psi, d_val

    def _encirclement_force(self, position: np.ndarray, target: Target, predicted_target: np.ndarray) -> np.ndarray:
        if not self.method.use_encirclement:
            return np.zeros(2)
        center = (1.0 - self.params.beta_lead) * target.position + self.params.beta_lead * predicted_target
        delta = position - center
        dist = norm(delta)
        return -self.params.k_r * (dist - self.params.r_c) * delta / (dist + EPS)

    def _angle_force(self, idx: int, positions: np.ndarray, target: Target, predicted_target: np.ndarray) -> np.ndarray:
        if not self.method.use_angle_spread or len(positions) < 3:
            return np.zeros(2)
        center = (1.0 - self.params.beta_lead) * target.position + self.params.beta_lead * predicted_target
        rel = positions - center[None, :]
        angles = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi)
        order = np.argsort(angles)
        rank = int(np.where(order == idx)[0][0])
        prev_idx = order[(rank - 1) % len(order)]
        next_idx = order[(rank + 1) % len(order)]
        theta_i = angles[idx]
        dtheta_plus = float((angles[next_idx] - theta_i) % (2 * np.pi))
        dtheta_minus = float((theta_i - angles[prev_idx]) % (2 * np.pi))
        radial = unit(positions[idx] - center)
        tangential = rotate90(radial)
        return self.params.k_angle * (dtheta_plus - dtheta_minus) * tangential

    def compute(self, positions: np.ndarray, target: Target, omega_prev: np.ndarray, obstacles: list[Obstacle], t: float, scenario: Scenario | None = None) -> dict[str, np.ndarray]:
        self.params_obstacles = obstacles
        n_agents = len(positions)
        predicted_target = self._predicted_target(target)
        neighbors = self._neighbor_sets(positions)
        u = np.zeros_like(positions)
        omega = np.zeros(n_agents)
        rho = np.zeros(n_agents)
        topo_gain = np.zeros(n_agents)
        psi_hat = np.zeros(n_agents)
        psi_tilde = np.zeros(n_agents)
        phi_vals = np.zeros(n_agents)
        nav_terms = np.zeros_like(positions)
        rep_terms = np.zeros_like(positions)
        curl_terms = np.zeros_like(positions)
        safe_terms = np.zeros_like(positions)
        topo_terms = np.zeros_like(positions)
        enc_terms = np.zeros_like(positions)
        angle_terms = np.zeros_like(positions)
        tensor_enc_scales = np.ones(n_agents)
        tensor_angle_scales = np.ones(n_agents)
        tensor_modulation_mats = np.repeat(np.eye(2)[None, :, :], n_agents, axis=0)
        xi_axes = np.repeat(np.array([[1.0, 0.0]], dtype=float), n_agents, axis=0)
        decision_z_values = np.zeros((n_agents, 4))
        coverage_aux_values = np.zeros(n_agents)
        nav_scales = np.ones(n_agents)
        safe_scales = np.ones(n_agents)
        directional_scales = np.ones(n_agents)
        guard_scales = np.ones(n_agents)
        enc_scales = np.ones(n_agents)
        angle_scales = np.ones(n_agents)
        xi_state_values = np.full((n_agents, 5), np.nan, dtype=float)
        env_axes = np.full((n_agents, 2), np.nan, dtype=float)
        form_axes = np.full((n_agents, 2), np.nan, dtype=float)
        tensor_axes = np.full((n_agents, 2), np.nan, dtype=float)
        env_pressures = np.full(n_agents, np.nan, dtype=float)
        env_anisotropies = np.full(n_agents, np.nan, dtype=float)
        form_anisotropies = np.full(n_agents, np.nan, dtype=float)
        mode_logits_values = np.full((n_agents, 4), np.nan, dtype=float)
        mode_weight_values = np.full((n_agents, 4), np.nan, dtype=float)
        corridor_preferred_velocities = np.zeros_like(positions)
        global_mode_logits = np.full(4, np.nan, dtype=float)
        global_mode_weights = np.full(4, np.nan, dtype=float)
        corridor_confidence = np.nan
        corridor_commitment = np.nan
        release_need = np.nan
        xi_feedback_crowding_mean = np.nan
        xi_feedback_crowding_peak = np.nan
        xi_feedback_slot_order_readiness = np.nan
        xi_feedback_lateral_order_readiness = np.nan
        xi_feedback_forward_room_mean = np.nan
        xi_feedback_obstacle_block_mean = np.nan
        xi_feedback_release_ready = np.nan
        guard_release_signal = np.nan
        queue_guard_decay = np.nan
        directional_release_signal = np.nan
        passage_push_strength = np.nan
        entry_progress = np.nan
        traverse_progress = np.nan
        exit_progress = np.nan
        forward_free_length = np.nan
        post_bottleneck_opening = np.nan
        front_blocker_bias = np.nan
        structured_energy_tank = clip01(self.transport_energy_tank / max(self.params.magnetic_tank_capacity, EPS))
        structured_energy_scene_signal = np.nan
        structured_energy_topology_signal = np.nan
        structured_energy_store = np.nan
        structured_energy_release = np.nan
        structured_energy_loss = np.nan
        agent_energy_budget = np.zeros(n_agents, dtype=float)
        structured_energy_approach_tank = np.nan
        structured_energy_transport_tank = np.nan
        structured_energy_convert = np.nan
        structured_energy_initial_ready = np.nan
        structured_energy_initial_debt = np.nan
        spring_chain_storage = np.nan
        spring_head_release = np.nan
        spring_rear_support = np.nan
        spring_link_compression = np.nan
        magnetic_transport_release_proxy = np.zeros(n_agents, dtype=float)
        unified_r = np.nan
        unified_w = np.nan
        unified_q = np.nan
        unified_a = np.nan
        unified_e = np.nan
        unified_entry_push = np.nan
        magnetic_drive_intent = np.zeros(n_agents, dtype=float)
        magnetic_release_intent = np.zeros(n_agents, dtype=float)
        magnetic_queue_scale = np.ones(n_agents, dtype=float)
        magnetic_effective_drive = np.zeros(n_agents, dtype=float)

        if self.method.baseline_mode == "orca_reactive":
            success_mode = scenario.success_mode if scenario is not None else "target_track"
            preferred_velocities = np.zeros_like(positions)
            for i in range(n_agents):
                preferred_velocities[i] = self._orca_preferred_velocity(i, positions, target, predicted_target, success_mode)
            u = np.zeros_like(positions)
            for i in range(n_agents):
                u[i] = self._orca_velocity(i, positions, preferred_velocities, neighbors, obstacles)
            self.prev_baseline_vel = u.copy()
            self.prev_output_vel = u.copy()
            env = np.array([self.params.env_flow(p, t) for p in positions])
            return {
                "u": u,
                "omega": omega,
                "rho": rho,
                "topo_gain": topo_gain,
                "psi_hat": psi_hat,
                "psi_tilde": psi_tilde,
                "phi": phi_vals,
                "env": env,
                "predicted_target": predicted_target,
                "xi_axis": xi_axes,
                "decision_z": decision_z_values,
                "coverage_aux": coverage_aux_values,
                "nav_scale": nav_scales,
                "safe_scale": safe_scales,
                "directional_scale": directional_scales,
                "guard_scale": guard_scales,
                "enc_scale": enc_scales,
                "angle_scale": angle_scales,
                "xi_state": xi_state_values,
                "env_axis": env_axes,
                "form_axis": form_axes,
                "tensor_axis": tensor_axes,
                "env_pressure": env_pressures,
                "env_anisotropy": env_anisotropies,
                "form_anisotropy": form_anisotropies,
                "mode_logits": mode_logits_values,
                "mode_weights": mode_weight_values,
                "directional_axis": np.array([1.0, 0.0], dtype=float),
                "corridor_activation": 0.0,
                "queue_activation": 0.0,
                "queue_mode": 0.0,
                "structure_lateral_width": 0.0,
                "structure_longitudinal_span": 0.0,
                "global_mode_logits": global_mode_logits,
                "global_mode_weights": global_mode_weights,
                "corridor_confidence": corridor_confidence,
                "corridor_commitment": corridor_commitment,
                "release_need": release_need,
                "xi_feedback_crowding_mean": xi_feedback_crowding_mean,
                "xi_feedback_crowding_peak": xi_feedback_crowding_peak,
                "xi_feedback_slot_order_readiness": xi_feedback_slot_order_readiness,
                "xi_feedback_lateral_order_readiness": xi_feedback_lateral_order_readiness,
                "xi_feedback_forward_room_mean": xi_feedback_forward_room_mean,
                "xi_feedback_obstacle_block_mean": xi_feedback_obstacle_block_mean,
                "xi_feedback_release_ready": xi_feedback_release_ready,
                "guard_release_signal": guard_release_signal,
                "queue_guard_decay": queue_guard_decay,
                "directional_release_signal": directional_release_signal,
                "passage_push_strength": passage_push_strength,
                "entry_progress": entry_progress,
                "traverse_progress": traverse_progress,
                "exit_progress": exit_progress,
                "forward_free_length": forward_free_length,
                "post_bottleneck_opening": post_bottleneck_opening,
                "front_blocker_bias": front_blocker_bias,
            }

        if self.method.baseline_mode == "cbf_qp_filter":
            success_mode = scenario.success_mode if scenario is not None else "target_track"
            nominal_velocities = np.zeros_like(positions)
            for i in range(n_agents):
                nominal_velocities[i] = self._cbf_nominal_velocity(i, positions, neighbors[i], target, predicted_target, success_mode)
            u = self._cbf_filter_velocities(positions, nominal_velocities, neighbors, obstacles)
            self.prev_baseline_vel = u.copy()
            self.prev_output_vel = u.copy()
            env = np.array([self.params.env_flow(p, t) for p in positions])
            return {
                "u": u,
                "omega": omega,
                "rho": rho,
                "topo_gain": topo_gain,
                "psi_hat": psi_hat,
                "psi_tilde": psi_tilde,
                "phi": phi_vals,
                "env": env,
                "predicted_target": predicted_target,
                "xi_axis": xi_axes,
                "decision_z": decision_z_values,
                "coverage_aux": coverage_aux_values,
                "nav_scale": nav_scales,
                "safe_scale": safe_scales,
                "directional_scale": directional_scales,
                "guard_scale": guard_scales,
                "enc_scale": enc_scales,
                "angle_scale": angle_scales,
                "xi_state": xi_state_values,
                "env_axis": env_axes,
                "form_axis": form_axes,
                "tensor_axis": tensor_axes,
                "env_pressure": env_pressures,
                "env_anisotropy": env_anisotropies,
                "form_anisotropy": form_anisotropies,
                "mode_logits": mode_logits_values,
                "mode_weights": mode_weight_values,
                "directional_axis": np.array([1.0, 0.0], dtype=float),
                "corridor_activation": 0.0,
                "queue_activation": 0.0,
                "queue_mode": 0.0,
                "structure_lateral_width": 0.0,
                "structure_longitudinal_span": 0.0,
                "global_mode_logits": global_mode_logits,
                "global_mode_weights": global_mode_weights,
                "corridor_confidence": corridor_confidence,
                "corridor_commitment": corridor_commitment,
                "release_need": release_need,
                "xi_feedback_crowding_mean": xi_feedback_crowding_mean,
                "xi_feedback_crowding_peak": xi_feedback_crowding_peak,
                "xi_feedback_slot_order_readiness": xi_feedback_slot_order_readiness,
                "xi_feedback_lateral_order_readiness": xi_feedback_lateral_order_readiness,
                "xi_feedback_forward_room_mean": xi_feedback_forward_room_mean,
                "xi_feedback_obstacle_block_mean": xi_feedback_obstacle_block_mean,
                "xi_feedback_release_ready": xi_feedback_release_ready,
                "guard_release_signal": guard_release_signal,
                "queue_guard_decay": queue_guard_decay,
                "directional_release_signal": directional_release_signal,
                "passage_push_strength": passage_push_strength,
                "entry_progress": entry_progress,
                "traverse_progress": traverse_progress,
                "exit_progress": exit_progress,
                "forward_free_length": forward_free_length,
                "post_bottleneck_opening": post_bottleneck_opening,
                "front_blocker_bias": front_blocker_bias,
            }

        if self.method.baseline_mode == "mpc_cbf_style":
            resolved_scenario = scenario or Scenario(
                key="baseline_mpc_fallback",
                title="Baseline MPC Fallback",
                description="Fallback scene wrapper for MPC baseline.",
                n_agents=n_agents,
                obstacles=obstacles,
                initial_positions=positions.copy(),
                target_fn=lambda _time: target,
                params=self.params,
                success_mode="target_track",
            )
            nominal_velocities = self._mpc_nominal_velocities(positions, neighbors, target, predicted_target, resolved_scenario, t)
            u = self._cbf_filter_velocities(positions, nominal_velocities, neighbors, obstacles)
            self.prev_baseline_vel = u.copy()
            self.prev_output_vel = u.copy()
            env = np.array([self.params.env_flow(p, t) for p in positions])
            return {
                "u": u,
                "omega": omega,
                "rho": rho,
                "topo_gain": topo_gain,
                "psi_hat": psi_hat,
                "psi_tilde": psi_tilde,
                "phi": phi_vals,
                "env": env,
                "predicted_target": predicted_target,
                "xi_axis": xi_axes,
                "decision_z": decision_z_values,
                "coverage_aux": coverage_aux_values,
                "nav_scale": nav_scales,
                "safe_scale": safe_scales,
                "directional_scale": directional_scales,
                "guard_scale": guard_scales,
                "enc_scale": enc_scales,
                "angle_scale": angle_scales,
                "xi_state": xi_state_values,
                "env_axis": env_axes,
                "form_axis": form_axes,
                "tensor_axis": tensor_axes,
                "env_pressure": env_pressures,
                "env_anisotropy": env_anisotropies,
                "form_anisotropy": form_anisotropies,
                "mode_logits": mode_logits_values,
                "mode_weights": mode_weight_values,
                "directional_axis": np.array([1.0, 0.0], dtype=float),
                "corridor_activation": 0.0,
                "queue_activation": 0.0,
                "queue_mode": 0.0,
                "structure_lateral_width": 0.0,
                "structure_longitudinal_span": 0.0,
                "global_mode_logits": global_mode_logits,
                "global_mode_weights": global_mode_weights,
                "corridor_confidence": corridor_confidence,
                "corridor_commitment": corridor_commitment,
                "release_need": release_need,
                "xi_feedback_crowding_mean": xi_feedback_crowding_mean,
                "xi_feedback_crowding_peak": xi_feedback_crowding_peak,
                "xi_feedback_slot_order_readiness": xi_feedback_slot_order_readiness,
                "xi_feedback_lateral_order_readiness": xi_feedback_lateral_order_readiness,
                "xi_feedback_forward_room_mean": xi_feedback_forward_room_mean,
                "xi_feedback_obstacle_block_mean": xi_feedback_obstacle_block_mean,
                "xi_feedback_release_ready": xi_feedback_release_ready,
                "guard_release_signal": guard_release_signal,
                "queue_guard_decay": queue_guard_decay,
                "directional_release_signal": directional_release_signal,
                "passage_push_strength": passage_push_strength,
                "entry_progress": entry_progress,
                "traverse_progress": traverse_progress,
                "exit_progress": exit_progress,
                "forward_free_length": forward_free_length,
                "post_bottleneck_opening": post_bottleneck_opening,
                "front_blocker_bias": front_blocker_bias,
            }

        for i, position in enumerate(positions):
            if self.method.traditional_apf:
                nav = -self.params.w_g * (position - target.position)
                rep = sum((self._traditional_repulsion(position, obs) for obs in obstacles), start=np.zeros(2))
                raw = (nav + rep) / self.params.gamma_d
                u[i] = smooth_bound(raw, self.params.u_max)
                omega[i] = omega_prev[i]
                rho[i] = 0.0
                topo_gain[i] = 0.0
                psi_hat[i] = 1.0
                psi_tilde[i] = 1.0
                continue

            if self.method.baseline_mode == "consensus_ring":
                nav = self.params.w_g * unit(predicted_target - position)
                rep, phi = self._bounded_repulsion(position)
                safe = self._safe_force(i, positions, neighbors[i])
                topo = self._consensus_force(i, positions, neighbors[i])
                enc = self._encirclement_force(position, target, predicted_target)
                raw = (nav + rep + safe + topo + enc) / self.params.gamma_d
                u[i] = smooth_bound(raw, self.params.u_max)
                omega[i] = phi
                rho[i] = 1.0
                topo_gain[i] = 1.0
                psi_hat[i] = 1.0
                psi_tilde[i] = 1.0
                phi_vals[i] = phi
                continue

            if self.method.baseline_mode == "leader_follower":
                lead = self._leader_follower_force(i, positions, target, predicted_target)
                rep, phi = self._bounded_repulsion(position)
                safe = self._safe_force(i, positions, neighbors[i])
                enc = 0.4 * self._encirclement_force(position, target, predicted_target)
                raw = (lead + rep + safe + enc) / self.params.gamma_d
                u[i] = smooth_bound(raw, self.params.u_max)
                omega[i] = phi
                rho[i] = 1.0
                topo_gain[i] = 1.0
                psi_hat[i] = 1.0
                psi_tilde[i] = 1.0
                phi_vals[i] = phi
                continue

            nav_dir = predicted_target - position
            if self.method.use_bounded_nav:
                nav = self.params.w_g * unit(nav_dir)
            else:
                nav = self.params.w_g * nav_dir

            rep, phi = self._bounded_repulsion(position) if self.method.use_bounded_repulsion else (sum((self._traditional_repulsion(position, obs) for obs in obstacles), start=np.zeros(2)), 0.0)
            curl = self._curl_term(i, position, rep, predicted_target)
            safe = self._safe_force(i, positions, neighbors[i])
            topo = self._topology_force(i, positions, neighbors[i])
            enc = self._encirclement_force(position, target, predicted_target)
            ang = self._angle_force(i, positions, target, predicted_target)
            nav_terms[i] = nav
            rep_terms[i] = rep
            curl_terms[i] = curl
            safe_terms[i] = safe
            topo_terms[i] = topo
            enc_terms[i] = enc
            angle_terms[i] = ang
            psi_hat[i], psi_tilde[i], d_val = self._local_psi(i, positions, target, neighbors[i], n_agents)
            alpha = min(1.0, self.params.dt / max(self.params.tau_omega, self.params.dt))
            omega[i] = clip01((1.0 - alpha) * omega_prev[i] + alpha * phi)
            phi_vals[i] = phi
            eta = 1.0 if not self.method.use_omega else self.params.eta_min + (1.0 - self.params.eta_min) * np.exp(-self.params.sigma_omega * omega[i])
            if self.method.force_rho_one:
                rho[i] = 1.0
            else:
                rho[i] = eta * d_val
            if self.method.force_rho_one:
                total_topo = topo
                topo_gain[i] = 1.0
            else:
                topo_floor = self.topo_floor_state[i]
                if self.params.topo_rho_floor > 0.0:
                    if omega[i] >= self.params.topo_floor_on_threshold:
                        topo_floor = self.params.topo_rho_floor
                    elif omega[i] <= self.params.topo_floor_off_threshold:
                        release_alpha = min(1.0, self.params.dt / max(self.params.topo_floor_release_tau, self.params.dt))
                        topo_floor = (1.0 - release_alpha) * topo_floor
                self.topo_floor_state[i] = topo_floor
                topo_scale = topo_floor + (1.0 - topo_floor) * rho[i]
                total_topo = topo_scale * topo
                topo_gain[i] = topo_scale
            if self.method.use_tensor_modulation:
                tensor_matrix, tensor_enc_scale, tensor_angle_scale, _env_pressure, _form_anisotropy, understanding_details = self._tensor_modulation_summary(
                    i,
                    positions,
                    neighbors[i],
                    position,
                    target,
                    predicted_target,
                )
                total_topo = tensor_matrix @ total_topo
                tensor_enc_scales[i] = tensor_enc_scale
                tensor_angle_scales[i] = tensor_angle_scale
            elif self.method.use_tensor_energy_polar:
                if self.method.use_global_phase_context:
                    env_matrix, env_axis, env_pressure, env_anisotropy = self._tensor_environment_matrix(position, predicted_target)
                    form_matrix, form_axis, form_anisotropy, _form_spread = self._tensor_formation_matrix(i, positions, neighbors[i], predicted_target)
                    del env_matrix, form_matrix, _form_spread
                    understanding_details = self._empty_understanding_details()
                    understanding_details.update(
                        {
                            "env_axis": env_axis,
                            "form_axis": form_axis,
                            "env_pressure": env_pressure,
                            "env_anisotropy": env_anisotropy,
                            "form_anisotropy": form_anisotropy,
                        }
                    )
                elif self.method.use_continuous_polar_modes:
                    tensor_matrix, tensor_enc_scale, tensor_angle_scale, _env_pressure, _form_anisotropy, understanding_details = self._continuous_polar_mode_summary(
                        i,
                        positions,
                        neighbors[i],
                        position,
                        target,
                        predicted_target,
                    )
                else:
                    tensor_matrix, tensor_enc_scale, tensor_angle_scale, _env_pressure, _form_anisotropy, understanding_details = self._tensor_energy_polar_summary(
                        i,
                        positions,
                        neighbors[i],
                        position,
                        target,
                        predicted_target,
                    )
                if not self.method.use_global_phase_context:
                    tensor_modulation_mats[i] = tensor_matrix
                    tensor_enc_scales[i] = tensor_enc_scale
                    tensor_angle_scales[i] = tensor_angle_scale
            else:
                understanding_details = self._empty_understanding_details()
            env_axes[i] = understanding_details["env_axis"]
            form_axes[i] = understanding_details["form_axis"]
            tensor_axes[i] = understanding_details["tensor_axis"]
            env_pressures[i] = float(understanding_details["env_pressure"])
            env_anisotropies[i] = float(understanding_details["env_anisotropy"])
            form_anisotropies[i] = float(understanding_details["form_anisotropy"])
            mode_logits_values[i] = understanding_details["mode_logits"]
            mode_weight_values[i] = understanding_details["mode_weights"]
            topo_terms[i] = total_topo

        global_phase_context = None
        if self.method.use_global_phase_context:
            global_phase_context = self._global_phase_context(
                positions,
                target,
                predicted_target,
                omega,
                env_pressures,
                env_anisotropies,
                form_anisotropies,
                scenario,
            )
            global_mode_logits = np.asarray(global_phase_context["mode_logits"], dtype=float)
            global_mode_weights = np.asarray(global_phase_context["mode_weights"], dtype=float)
            corridor_confidence = float(global_phase_context["corridor_confidence"])
            corridor_commitment = float(global_phase_context["corridor_commitment"])
            release_need = float(global_phase_context["release_need"])
            xi_feedback_crowding_mean = float(global_phase_context["xi_feedback_crowding_mean"])
            xi_feedback_crowding_peak = float(global_phase_context["xi_feedback_crowding_peak"])
            xi_feedback_slot_order_readiness = float(global_phase_context["xi_feedback_slot_order_readiness"])
            xi_feedback_lateral_order_readiness = float(global_phase_context["xi_feedback_lateral_order_readiness"])
            xi_feedback_forward_room_mean = float(global_phase_context["xi_feedback_forward_room_mean"])
            xi_feedback_obstacle_block_mean = float(global_phase_context["xi_feedback_obstacle_block_mean"])
            xi_feedback_release_ready = float(global_phase_context["xi_feedback_release_ready"])
            entry_progress = float(global_phase_context["entry_progress"])
            traverse_progress = float(global_phase_context["traverse_progress"])
            exit_progress = float(global_phase_context["exit_progress"])
            forward_free_length = float(global_phase_context["forward_free_length"])
            post_bottleneck_opening = float(global_phase_context["post_bottleneck_opening"])
            front_blocker_bias = float(global_phase_context["front_blocker_bias"])
            if self.method.use_structured_energy_tank:
                energy_signals = self._update_structured_energy_tank(global_phase_context)
                structured_energy_tank = float(energy_signals["tank"])
                structured_energy_scene_signal = float(energy_signals["scene_signal"])
                structured_energy_topology_signal = float(energy_signals["topology_signal"])
                structured_energy_store = float(energy_signals["store"])
                structured_energy_release = float(energy_signals["release"])
                structured_energy_loss = float(energy_signals["loss"])
            elif self.method.use_dual_stage_energy_tank:
                energy_signals = self._update_dual_stage_energy_tanks(global_phase_context)
                structured_energy_tank = float(energy_signals["transport_tank"])
                structured_energy_scene_signal = float(energy_signals["scene_signal"])
                structured_energy_topology_signal = float(energy_signals["topology_signal"])
                structured_energy_store = float(energy_signals["transport_store"])
                structured_energy_release = float(energy_signals["transport_release"])
                structured_energy_loss = float(energy_signals["transport_loss"])
                structured_energy_approach_tank = float(energy_signals["approach_tank"])
                structured_energy_transport_tank = float(energy_signals["transport_tank"])
                structured_energy_convert = float(energy_signals["convert"])
                structured_energy_initial_ready = float(energy_signals["initial_ready"])
                structured_energy_initial_debt = float(energy_signals["initial_debt"])

        directional_force, queue_mode = self._directional_reconfigure_force(positions, target, predicted_target, omega)
        queue_guard_force = self._queue_spacing_guard_force(positions, target, predicted_target, queue_mode)
        if self.method.use_global_phase_context and global_phase_context is not None:
            forward_free_norm = clip01(forward_free_length / max(2.0 * self.params.r0, EPS))
            guard_release_signal = clip01(
                0.35 * corridor_commitment
                + 0.25 * release_need
                + 0.25 * xi_feedback_release_ready
                + 0.15 * traverse_progress
            )
            queue_guard_decay = float(
                np.clip(
                    1.0 - 0.7 * corridor_confidence * guard_release_signal,
                    0.2,
                    1.0,
                )
            )
            queue_guard_force = queue_guard_decay * queue_guard_force
            directional_release_signal = clip01(
                0.4 * corridor_commitment
                + 0.25 * xi_feedback_release_ready
                + 0.2 * traverse_progress
                + 0.15 * forward_free_norm
            )
            axis = np.asarray(global_phase_context["axis"], dtype=float)
            axis = unit(axis)
            if norm(axis) > EPS:
                parallel_component = np.outer(directional_force @ axis, axis)
                lateral_component = directional_force - parallel_component
                parallel_decay = float(np.clip(1.0 - 0.45 * corridor_confidence * directional_release_signal, 0.35, 1.0))
                lateral_decay = float(np.clip(1.0 - 0.8 * corridor_confidence * directional_release_signal, 0.15, 1.0))
                passage_push_strength = 0.24 * corridor_confidence * directional_release_signal * (0.45 + 0.55 * forward_free_norm)
                directional_force = (
                    parallel_decay * parallel_component
                    + lateral_decay * lateral_component
                    + passage_push_strength * axis[None, :]
                )
        enc_scale = 1.0 - 0.85 * queue_mode
        angle_scale = 1.0 - 0.5 * queue_mode
        polar_axis = self._directional_axis(positions, target, predicted_target)
        center = np.mean(positions, axis=0)
        perp_axis = rotate90(polar_axis)
        parallel_coords = np.array([float((position - center) @ polar_axis) for position in positions])
        lateral_coords = np.array([float((position - center) @ perp_axis) for position in positions])
        lateral_width = float(np.max(lateral_coords) - np.min(lateral_coords)) if len(lateral_coords) else 0.0
        longitudinal_span = float(np.max(parallel_coords) - np.min(parallel_coords)) if len(parallel_coords) else 0.0
        corridor_activation, queue_activation = self._directional_activation(omega, target)
        if global_phase_context is not None:
            polar_axis = np.asarray(global_phase_context["axis"], dtype=float)
            center = np.asarray(global_phase_context["center"], dtype=float)
            perp_axis = np.asarray(global_phase_context["perp_axis"], dtype=float)
            parallel_coords = np.asarray(global_phase_context["parallel_coords"], dtype=float)
            lateral_coords = np.asarray(global_phase_context["lateral_coords"], dtype=float)
            lateral_width = float(global_phase_context["lateral_width"])
            longitudinal_span = float(global_phase_context["longitudinal_span"])
            corridor_activation = float(global_phase_context["corridor_activation"])
            queue_activation = float(global_phase_context["queue_activation"])
        polar_activation = max(corridor_activation, queue_activation)
        matrix_activation = max(corridor_activation, queue_activation)
        matrix_transform = self._directional_matrix(
            polar_axis,
            1.0 + matrix_activation * (self.params.matrix_parallel_gain - 1.0),
            1.0 - matrix_activation * (1.0 - self.params.matrix_perp_gain),
        )
        if self.method.use_temporal_magnetic_release:
            self.magnetic_release_prev = self.magnetic_release_state.copy()

        propagated_entry_budgets = np.zeros(n_agents, dtype=float)
        precompute_xi_global = (self.method.use_entry_energy_propagation or self.method.use_energy_global_quota_redistribution or self.method.use_unified_state_entry_push)
        if precompute_xi_global and self.method.use_global_phase_context and global_phase_context is not None:
            for i in range(n_agents):
                tensor_matrix, tensor_enc_scale, tensor_angle_scale, _env_pressure, _form_anisotropy, understanding_details = self._global_phase_guided_tensor_summary(
                    i,
                    positions,
                    neighbors[i],
                    positions[i],
                    target,
                    predicted_target,
                    global_phase_context,
                )
                tensor_modulation_mats[i] = tensor_matrix
                tensor_enc_scales[i] = tensor_enc_scale
                tensor_angle_scales[i] = tensor_angle_scale
                env_axes[i] = understanding_details["env_axis"]
                form_axes[i] = understanding_details["form_axis"]
                tensor_axes[i] = understanding_details["tensor_axis"]
                env_pressures[i] = float(understanding_details["env_pressure"])
                env_anisotropies[i] = float(understanding_details["env_anisotropy"])
                form_anisotropies[i] = float(understanding_details["form_anisotropy"])
                mode_logits_values[i] = understanding_details["mode_logits"]
                mode_weight_values[i] = understanding_details["mode_weights"]
                xi_axes[i] = np.asarray(global_phase_context["axis"], dtype=float)
                xi_state_values[i] = self._local_role_state(i, positions, neighbors[i], global_phase_context)
                decision_z_values[i], coverage_aux_values[i] = self._local_role_decision_coordinates(
                    omega[i],
                    psi_tilde[i],
                    xi_state_values[i],
                    global_phase_context,
                )
                (
                    nav_scales[i],
                    safe_scales[i],
                    directional_scales[i],
                    guard_scales[i],
                    enc_scales[i],
                    angle_scales[i],
                ) = self._decision_module_scales(decision_z_values[i], coverage_aux_values[i])
            if self.method.use_entry_energy_propagation:
                propagated_entry_budgets = self._entry_energy_propagation_budgets(global_phase_context, xi_state_values)
            elif self.method.use_energy_global_quota_redistribution:
                propagated_entry_budgets = self._global_quota_xi_redistribution_budgets(global_phase_context, xi_state_values)
            if self.method.use_dual_stage_energy_tank:
                spring_chain = self._update_spring_chain_observables(global_phase_context, xi_state_values)
                spring_chain_storage = float(spring_chain["spring_chain_storage"])
                spring_head_release = float(spring_chain["spring_head_release"])
                spring_rear_support = float(spring_chain["spring_rear_support"])
                spring_link_compression = float(spring_chain["spring_link_compression"])

        for i in range(n_agents):
            if self.method.traditional_apf or self.method.baseline_mode in {"consensus_ring", "leader_follower"}:
                continue
            topo_term = topo_terms[i]
            if self.method.use_matrix_modulation:
                topo_term = matrix_transform @ topo_term
            if self.method.use_fusion_attention:
                raw = self._fusion_force(
                    i,
                    positions,
                    neighbors[i],
                    target,
                    predicted_target,
                    omega[i],
                    rho[i],
                    phi_vals[i],
                    nav_terms[i],
                    rep_terms[i],
                    safe_terms[i],
                    topo_term,
                    enc_scale * enc_terms[i],
                    angle_scale * angle_terms[i],
                    curl_terms[i],
                    directional_force[i],
                    queue_guard_force[i],
                    corridor_activation,
                    queue_activation,
                    lateral_width,
                    longitudinal_span,
                    lateral_coords[i],
                ) / self.params.gamma_d
            elif self.method.use_self_localization_state:
                if precompute_xi_global and self.method.use_global_phase_context and global_phase_context is not None:
                    xi_axis = xi_axes[i]
                    xi_state = xi_state_values[i]
                    decision_z = decision_z_values[i]
                    coverage_aux = coverage_aux_values[i]
                    nav_scale = nav_scales[i]
                    safe_scale = safe_scales[i]
                    directional_scale = directional_scales[i]
                    guard_scale = guard_scales[i]
                    enc_scale = enc_scales[i]
                    angle_scale = angle_scales[i]
                elif self.method.use_global_phase_context and global_phase_context is not None:
                    tensor_matrix, tensor_enc_scale, tensor_angle_scale, _env_pressure, _form_anisotropy, understanding_details = self._global_phase_guided_tensor_summary(
                        i,
                        positions,
                        neighbors[i],
                        positions[i],
                        target,
                        predicted_target,
                        global_phase_context,
                    )
                    tensor_modulation_mats[i] = tensor_matrix
                    tensor_enc_scales[i] = tensor_enc_scale
                    tensor_angle_scales[i] = tensor_angle_scale
                    env_axes[i] = understanding_details["env_axis"]
                    form_axes[i] = understanding_details["form_axis"]
                    tensor_axes[i] = understanding_details["tensor_axis"]
                    env_pressures[i] = float(understanding_details["env_pressure"])
                    env_anisotropies[i] = float(understanding_details["env_anisotropy"])
                    form_anisotropies[i] = float(understanding_details["form_anisotropy"])
                    mode_logits_values[i] = understanding_details["mode_logits"]
                    mode_weight_values[i] = understanding_details["mode_weights"]
                    xi_axis = np.asarray(global_phase_context["axis"], dtype=float)
                    xi_state = self._local_role_state(i, positions, neighbors[i], global_phase_context)
                    decision_z, coverage_aux = self._local_role_decision_coordinates(
                        omega[i],
                        psi_tilde[i],
                        xi_state,
                        global_phase_context,
                    )
                else:
                    xi_axis = self._principal_axis(tensor_modulation_mats[i], polar_axis)
                    xi_state = self._self_localization_state(i, positions, neighbors[i], xi_axis, center)
                    decision_z, coverage_aux = self._decision_coordinates(
                        omega[i],
                        psi_tilde[i],
                        xi_state,
                        corridor_activation,
                        queue_activation,
                    )
                nav_scale, safe_scale, directional_scale, guard_scale, enc_scale, angle_scale = self._decision_module_scales(decision_z, coverage_aux)
                xi_axes[i] = xi_axis
                decision_z_values[i] = decision_z
                coverage_aux_values[i] = coverage_aux
                nav_scales[i] = nav_scale
                safe_scales[i] = safe_scale
                directional_scales[i] = directional_scale
                guard_scales[i] = guard_scale
                enc_scales[i] = enc_scale
                angle_scales[i] = angle_scale
                xi_state_values[i] = xi_state
                if self.method.use_global_phase_context and global_phase_context is not None:
                    forward_free_norm = clip01(float(global_phase_context["forward_free_length"]) / max(2.0 * self.params.r0, EPS))
                    corridor_stage = clip01(float(global_phase_context["corridor_confidence"]) * max(float(global_phase_context["corridor_commitment"]), directional_release_signal))
                    nav_boost = 1.0 + 1.35 * corridor_stage * (0.45 + 0.55 * forward_free_norm)
                    safe_decay = float(np.clip(1.0 - 0.3 * corridor_stage, 0.5, 1.0))
                    topo_decay = float(np.clip(1.0 - 0.8 * corridor_stage, 0.12, 1.0))
                    enc_decay = float(np.clip(1.0 - 0.55 * corridor_stage, 0.2, 1.0))
                    directional_residual_gain = float(np.clip(1.0 - 0.4 * corridor_stage, 0.35, 1.0))
                    guard_residual_gain = float(np.clip(1.0 - 0.7 * corridor_stage, 0.15, 1.0))
                    passage_axis = unit(np.asarray(global_phase_context["axis"], dtype=float))
                    extra_passage_drive = np.zeros(2, dtype=float)
                    if norm(passage_axis) > EPS:
                        extra_passage_drive = 0.32 * corridor_stage * (0.35 + 0.65 * forward_free_norm) * passage_axis
                    conservative = self._energy_flow_force(
                        nav_boost * nav_scale * nav_terms[i],
                        rep_terms[i],
                        safe_decay * safe_scale * safe_terms[i],
                        topo_decay * topo_term,
                        enc_decay * enc_scale * tensor_enc_scales[i] * enc_terms[i],
                        angle_scale * tensor_angle_scales[i] * angle_terms[i],
                        omega[i],
                        rho[i],
                        max(corridor_activation, queue_activation),
                    )
                else:
                    directional_residual_gain = 1.0
                    guard_residual_gain = 1.0
                    extra_passage_drive = np.zeros(2, dtype=float)
                    conservative = self._energy_flow_force(
                        nav_scale * nav_terms[i],
                        rep_terms[i],
                        safe_scale * safe_terms[i],
                        topo_term,
                        enc_scale * tensor_enc_scales[i] * enc_terms[i],
                        angle_scale * tensor_angle_scales[i] * angle_terms[i],
                        omega[i],
                        rho[i],
                        max(corridor_activation, queue_activation),
                    )
                magnetic_force = np.zeros(2, dtype=float)
                magnetic_drive_force = np.zeros(2, dtype=float)
                queue_occupancy_force = np.zeros(2, dtype=float)
                queue_drive_scale = 1.0
                queue_push_force = np.zeros(2, dtype=float)
                leader_pull_force = np.zeros(2, dtype=float)
                soft_curl = curl_terms[i]
                if (
                    self.method.use_magnetic_corridor_adapter
                    and global_phase_context is not None
                    and scenario is not None
                    and scenario.success_mode == "corridor_pass"
                ):
                    if self.method.use_temporal_magnetic_release:
                        magnetic_force = self._xi_temporal_magnetic_corridor_force(i, positions, neighbors[i], global_phase_context, xi_state)
                    elif self.method.use_xi_magnetic_polarity:
                        magnetic_force = self._xi_magnetic_corridor_force(i, positions, neighbors[i], global_phase_context, xi_state)
                    else:
                        magnetic_force = self._magnetic_corridor_force(i, positions, neighbors[i], global_phase_context, xi_state)
                    magnetic_drive_force, magnetic_drive_intent[i], magnetic_release_intent[i] = self._magnetic_drive_force(
                        i,
                        nav_terms[i],
                        global_phase_context,
                        xi_state,
                    )
                    queue_occupancy_force, queue_drive_scale = self._soft_queue_occupancy_force(i, global_phase_context, xi_state)
                    queue_push_force, queue_push_drive_scale = self._soft_queue_push_force(i, global_phase_context, xi_state)
                    leader_pull_force, leader_drive_scale = self._soft_leader_pull_force(i, global_phase_context, xi_state)
                    queue_drive_scale = min(queue_drive_scale, queue_push_drive_scale)
                    queue_drive_scale = min(queue_drive_scale, leader_drive_scale)
                    magnetic_queue_scale[i] = queue_drive_scale
                    magnetic_drive_force = queue_drive_scale * magnetic_drive_force
                    if self.method.use_aggressive_magnetic_soft_drive:
                        soft_curl = self.params.magnetic_soft_curl_boost * curl_terms[i]
                    if self.method.use_high_energy_polarity_drive:
                        soft_curl = self.params.magnetic_high_curl_boost * curl_terms[i]
                    if self.method.use_adaptive_xi_energy_drive:
                        _slot_error, lateral_offset, gap_balance, obs_bias, crowding = [float(value) for value in xi_state]
                        obstacle_clearance = clip01(0.5 * (obs_bias + 1.0))
                        forward_room = clip01(0.5 * (gap_balance + 1.0))
                        xi_alignment = clip01(1.0 - 0.45 * abs(lateral_offset))
                        adaptive_curl_gate = clip01(
                            0.28 * obstacle_clearance
                            + 0.22 * forward_room
                            + 0.2 * xi_alignment
                            + 0.18 * clip01(float(global_phase_context.get("corridor_commitment", 0.0)))
                            - 0.24 * crowding
                        )
                        curl_gain = 0.7 + (self.params.magnetic_adaptive_curl_ceiling - 0.7) * adaptive_curl_gate
                        soft_curl = curl_gain * curl_terms[i]
                    if self.method.use_structured_energy_tank:
                        local_budget, drive_gain, curl_gain = self._structured_energy_agent_budget(i, global_phase_context, xi_state, queue_drive_scale)
                        agent_energy_budget[i] = local_budget
                        magnetic_drive_force = drive_gain * magnetic_drive_force
                        soft_curl = curl_gain * curl_terms[i]
                    elif self.method.use_dual_stage_energy_tank:
                        local_budget, drive_gain, curl_gain = self._dual_stage_agent_budget(i, global_phase_context, xi_state, queue_drive_scale, propagated_entry_budgets[i])
                        agent_energy_budget[i] = local_budget
                        magnetic_drive_force = drive_gain * magnetic_drive_force
                        soft_curl = curl_gain * curl_terms[i]
                        transport_release_drive, magnetic_transport_release_proxy[i] = self._transport_tank_release_drive(
                            i,
                            global_phase_context,
                            xi_state,
                            queue_drive_scale,
                            propagated_entry_budgets[i],
                        )
                        magnetic_drive_force = magnetic_drive_force + transport_release_drive
                    if self.method.use_unified_state_entry_push and np.isfinite(unified_r) and unified_r < 0.0:
                        magnetic_drive_force = unified_entry_push_vec + 0.15 * magnetic_drive_force
                    magnetic_effective_drive[i] = norm(magnetic_drive_force)
                if self.method.use_pure_magnetic_corridor_control and norm(magnetic_force) > EPS:
                    if self.method.use_magnetic_drive_energy:
                        raw = (magnetic_force + magnetic_drive_force + queue_occupancy_force + queue_push_force + leader_pull_force + soft_curl) / self.params.gamma_d
                    else:
                        raw = magnetic_force / self.params.gamma_d
                else:
                    raw = (
                        tensor_modulation_mats[i] @ conservative
                        + self.params.tensor_shape_residual_gain * (directional_residual_gain * directional_scale * directional_force[i] + guard_residual_gain * guard_scale * queue_guard_force[i])
                        + extra_passage_drive
                        + curl_terms[i]
                    ) / self.params.gamma_d
                    if norm(magnetic_force) > EPS:
                        raw = raw + (magnetic_force + magnetic_drive_force + queue_occupancy_force + queue_push_force + leader_pull_force) / self.params.gamma_d
            elif self.method.use_tensor_energy_polar:
                conservative = self._energy_flow_force(
                    nav_terms[i],
                    rep_terms[i],
                    safe_terms[i],
                    topo_term,
                    tensor_enc_scales[i] * enc_terms[i],
                    tensor_angle_scales[i] * angle_terms[i],
                    omega[i],
                    rho[i],
                    max(corridor_activation, queue_activation),
                )
                raw = (
                    tensor_modulation_mats[i] @ conservative
                    + self.params.tensor_shape_residual_gain * (directional_force[i] + queue_guard_force[i])
                    + curl_terms[i]
                ) / self.params.gamma_d
            elif self.method.use_energy_flow:
                conservative = self._energy_flow_force(
                    nav_terms[i],
                    rep_terms[i],
                    safe_terms[i],
                    topo_term,
                    enc_terms[i],
                    angle_terms[i],
                    omega[i],
                    rho[i],
                    max(corridor_activation, queue_activation),
                )
                raw = (conservative + directional_force[i] + queue_guard_force[i] + curl_terms[i]) / self.params.gamma_d
            elif self.method.use_projection_priority:
                raw = self._projection_priority_force(
                    nav_terms[i],
                    rep_terms[i],
                    curl_terms[i],
                    safe_terms[i],
                    topo_term,
                    enc_scale * enc_terms[i],
                    angle_scale * angle_terms[i],
                    directional_force[i],
                    queue_guard_force[i],
                ) / self.params.gamma_d
            else:
                enc_term = enc_scale * enc_terms[i]
                angle_term = angle_scale * angle_terms[i]
                if self.method.use_tensor_modulation:
                    enc_term = tensor_enc_scales[i] * enc_terms[i]
                    angle_term = tensor_angle_scales[i] * angle_terms[i]
                polar_force = self._polar_kernel_force(i, positions, neighbors[i], polar_axis, polar_activation)
                raw = (
                    nav_terms[i]
                    + rep_terms[i]
                    + curl_terms[i]
                    + safe_terms[i]
                    + topo_term
                    + directional_force[i]
                    + polar_force
                    + enc_term
                    + angle_term
                ) / self.params.gamma_d
            u[i] = smooth_bound(raw, self.params.u_max)
            corridor_preferred_velocities[i] = u[i]

        if self.method.use_global_phase_context and self.method.use_self_localization_state:
            self.prev_xi_feedback = self._aggregate_xi_feedback(xi_state_values)

        if global_phase_context is not None and (self.method.use_corridor_projection_low_level or self.method.use_corridor_qp_low_level):
            stage_strength = self._corridor_low_level_stage_strength(global_phase_context)
            if (scenario is None or scenario.success_mode == "corridor_pass") and stage_strength > 0.05:
                if self.method.use_corridor_projection_low_level:
                    u = self._corridor_projection_filter(positions, corridor_preferred_velocities, global_phase_context)
                elif self.method.use_corridor_qp_low_level:
                    u = self._corridor_qp_filter(positions, corridor_preferred_velocities, global_phase_context)

        unified_state = self._unified_state_observables(
            global_phase_context,
            xi_axes,
            xi_state_values,
            0.0 if np.isnan(structured_energy_transport_tank) else structured_energy_transport_tank,
            0.0 if np.isnan(structured_energy_approach_tank) else structured_energy_approach_tank,
            scenario,
        )
        unified_r = unified_state["r"]
        unified_w = unified_state["w"]
        unified_q = unified_state["q"]
        unified_a = unified_state["a"]
        unified_e = unified_state["e"]
        unified_entry_push_vec = np.zeros(2, dtype=float)
        if self.method.use_unified_state_entry_push and global_phase_context is not None:
            unified_entry_push_vec = self._unified_state_entry_push(unified_state, np.asarray(global_phase_context["axis"], dtype=float))
        unified_entry_push = float(norm(unified_entry_push_vec))

        self.prev_output_vel = u.copy()
        env = np.array([self.params.env_flow(p, t) for p in positions])
        return {
            "u": u,
            "omega": omega,
            "rho": rho,
            "topo_gain": topo_gain,
            "psi_hat": psi_hat,
            "psi_tilde": psi_tilde,
            "phi": phi_vals,
            "env": env,
            "predicted_target": predicted_target,
            "xi_axis": xi_axes,
            "decision_z": decision_z_values,
            "coverage_aux": coverage_aux_values,
            "nav_scale": nav_scales,
            "safe_scale": safe_scales,
            "directional_scale": directional_scales,
            "guard_scale": guard_scales,
            "enc_scale": enc_scales,
            "angle_scale": angle_scales,
            "xi_state": xi_state_values,
            "env_axis": env_axes,
            "form_axis": form_axes,
            "tensor_axis": tensor_axes,
            "env_pressure": env_pressures,
            "env_anisotropy": env_anisotropies,
            "form_anisotropy": form_anisotropies,
            "mode_logits": mode_logits_values,
            "mode_weights": mode_weight_values,
            "directional_axis": polar_axis,
            "corridor_activation": corridor_activation,
            "queue_activation": queue_activation,
            "queue_mode": queue_mode,
            "structure_lateral_width": lateral_width,
            "structure_longitudinal_span": longitudinal_span,
            "global_mode_logits": global_mode_logits,
            "global_mode_weights": global_mode_weights,
            "corridor_confidence": corridor_confidence,
            "corridor_commitment": corridor_commitment,
            "release_need": release_need,
            "xi_feedback_crowding_mean": xi_feedback_crowding_mean,
            "xi_feedback_crowding_peak": xi_feedback_crowding_peak,
            "xi_feedback_slot_order_readiness": xi_feedback_slot_order_readiness,
            "xi_feedback_lateral_order_readiness": xi_feedback_lateral_order_readiness,
            "xi_feedback_forward_room_mean": xi_feedback_forward_room_mean,
            "xi_feedback_obstacle_block_mean": xi_feedback_obstacle_block_mean,
            "xi_feedback_release_ready": xi_feedback_release_ready,
            "guard_release_signal": guard_release_signal,
            "queue_guard_decay": queue_guard_decay,
            "directional_release_signal": directional_release_signal,
            "passage_push_strength": passage_push_strength,
            "entry_progress": entry_progress,
            "traverse_progress": traverse_progress,
            "exit_progress": exit_progress,
            "forward_free_length": forward_free_length,
            "obstacle_pressure": float(global_phase_context.get("obstacle_pressure", np.nan)) if global_phase_context is not None else np.nan,
            "teammate_pressure": float(global_phase_context.get("teammate_pressure", np.nan)) if global_phase_context is not None else np.nan,
            "post_bottleneck_opening": post_bottleneck_opening,
            "front_blocker_bias": front_blocker_bias,
            "structured_energy_tank": structured_energy_tank,
            "structured_energy_scene_signal": structured_energy_scene_signal,
            "structured_energy_topology_signal": structured_energy_topology_signal,
            "structured_energy_store": structured_energy_store,
            "structured_energy_release": structured_energy_release,
            "structured_energy_loss": structured_energy_loss,
            "agent_energy_budget": agent_energy_budget,
            "structured_energy_approach_tank": structured_energy_approach_tank,
            "structured_energy_transport_tank": structured_energy_transport_tank,
            "structured_energy_convert": structured_energy_convert,
            "structured_energy_initial_ready": structured_energy_initial_ready,
            "structured_energy_initial_debt": structured_energy_initial_debt,
            "spring_chain_storage": spring_chain_storage,
            "spring_head_release": spring_head_release,
            "spring_rear_support": spring_rear_support,
            "spring_link_compression": spring_link_compression,
            "magnetic_transport_release_proxy": magnetic_transport_release_proxy,
            "magnetic_drive_intent": magnetic_drive_intent,
            "magnetic_release_intent": magnetic_release_intent,
            "magnetic_queue_scale": magnetic_queue_scale,
            "magnetic_effective_drive": magnetic_effective_drive,
            "unified_r": unified_r,
            "unified_w": unified_w,
            "unified_q": unified_q,
            "unified_a": unified_a,
            "unified_e": unified_e,
            "unified_entry_push": unified_entry_push,
        }


def with_horizon(params: Params, horizon: float) -> Params:
    return replace(params, horizon=horizon)
