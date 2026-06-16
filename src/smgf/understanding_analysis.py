from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import METHOD_LIBRARY, SCENARIOS, run_scene_trial


MODE_COLUMNS = ["encircle", "corridor", "queue", "recover"]
DECISION_COLUMNS = ["progress", "caution", "reshape", "spacing"]
SCALE_COLUMNS = ["nav_scale", "safe_scale", "directional_scale", "guard_scale", "enc_scale", "angle_scale"]
XI_COLUMNS = ["front_rank", "lateral", "shell", "clearance_bias", "crowding"]


def _result_time_vector(result: dict) -> np.ndarray:
    dt = result["params"].dt
    return np.arange(len(result["positions_hist"])) * dt


def _nanmean(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float)
    if flat.size == 0 or np.all(np.isnan(flat)):
        return float("nan")
    return float(np.nanmean(flat))


def _nanmax(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float)
    if flat.size == 0 or np.all(np.isnan(flat)):
        return float("nan")
    return float(np.nanmax(flat))


def _alignment_mean(axes: np.ndarray, reference: np.ndarray) -> float:
    if len(axes) == 0:
        return float("nan")
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm <= 1e-9:
        return float("nan")
    ref = reference / ref_norm
    values = []
    for axis in axes:
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-9 or np.any(np.isnan(axis)):
            continue
        values.append(abs(float((axis / axis_norm) @ ref)))
    if not values:
        return float("nan")
    return float(np.mean(values))


def _mode_entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0 or np.any(np.isnan(weights)):
        return float("nan")
    clipped = np.clip(weights, 1e-9, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _dominant_mode(weights: np.ndarray) -> str:
    weights = np.asarray(weights, dtype=float)
    if weights.size != len(MODE_COLUMNS) or np.any(np.isnan(weights)):
        return ""
    return MODE_COLUMNS[int(np.argmax(weights))]


def _dominant_mode_switch_count(labels: list[str]) -> int:
    filtered = [label for label in labels if label]
    if not filtered:
        return 0
    switches = 0
    previous = filtered[0]
    for label in filtered[1:]:
        if label != previous:
            switches += 1
            previous = label
    return switches


def _late_mask(length: int) -> np.ndarray:
    window = max(1, length // 4)
    mask = np.zeros(length, dtype=bool)
    mask[-window:] = True
    return mask


def _positive_derivative(values: np.ndarray, dt: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    deriv = np.full_like(arr, np.nan)
    finite = np.isfinite(arr)
    if len(arr) <= 1 or dt <= 0.0:
        return deriv
    for idx in range(1, len(arr)):
        if finite[idx] and finite[idx - 1]:
            deriv[idx] = max((arr[idx] - arr[idx - 1]) / dt, 0.0)
    if len(arr) > 1:
        deriv[0] = deriv[1]
    return deriv


def _corridor_post_exit_mask(scene, agents_past_exit_ratio: np.ndarray) -> np.ndarray:
    if scene.corridor_exit_x is None:
        return np.zeros_like(agents_past_exit_ratio, dtype=bool)
    return agents_past_exit_ratio >= 0.33


def _agent_understanding_dataframe(result: dict, method_key: str) -> pd.DataFrame:
    time = _result_time_vector(result)
    positions_hist = result["positions_hist"]
    u_hist = result["u_hist"]
    omega_hist = result["omega_hist"]
    rho_hist = result["rho_hist"]
    topo_gain_hist = result["topo_gain_hist"]
    psi_hist = result["psi_hist"]
    phi_hist = result["phi_hist"]
    xi_axis_hist = result["xi_axis_hist"]
    xi_state_hist = result["xi_state_hist"]
    decision_z_hist = result["decision_z_hist"]
    coverage_aux_hist = result["coverage_aux_hist"]
    nav_scale_hist = result["nav_scale_hist"]
    safe_scale_hist = result["safe_scale_hist"]
    directional_scale_hist = result["directional_scale_hist"]
    guard_scale_hist = result["guard_scale_hist"]
    enc_scale_hist = result["enc_scale_hist"]
    angle_scale_hist = result["angle_scale_hist"]
    env_axis_hist = result["env_axis_hist"]
    form_axis_hist = result["form_axis_hist"]
    tensor_axis_hist = result["tensor_axis_hist"]
    env_pressure_hist = result["env_pressure_hist"]
    env_anisotropy_hist = result["env_anisotropy_hist"]
    form_anisotropy_hist = result["form_anisotropy_hist"]
    mode_logits_hist = result["mode_logits_hist"]
    mode_weights_hist = result["mode_weights_hist"]
    magnetic_drive_intent_hist = result.get("magnetic_drive_intent_hist")
    magnetic_release_intent_hist = result.get("magnetic_release_intent_hist")
    magnetic_queue_scale_hist = result.get("magnetic_queue_scale_hist")
    magnetic_effective_drive_hist = result.get("magnetic_effective_drive_hist")
    magnetic_transport_release_proxy_hist = result.get("magnetic_transport_release_proxy_hist")
    spring_chain_storage_hist = result.get("spring_chain_storage_hist")
    spring_head_release_hist = result.get("spring_head_release_hist")
    spring_rear_support_hist = result.get("spring_rear_support_hist")
    spring_link_compression_hist = result.get("spring_link_compression_hist")

    rows: list[dict] = []
    for step_idx, t in enumerate(time):
        for agent_idx in range(positions_hist.shape[1]):
            row = {
                "scene": result["scene"].key,
                "method": method_key,
                "seed": result["seed"],
                "step": step_idx,
                "time": float(t),
                "agent": agent_idx,
                "x": float(positions_hist[step_idx, agent_idx, 0]),
                "y": float(positions_hist[step_idx, agent_idx, 1]),
                "ux": float(u_hist[step_idx, agent_idx, 0]),
                "uy": float(u_hist[step_idx, agent_idx, 1]),
                "omega": float(omega_hist[step_idx, agent_idx]),
                "rho": float(rho_hist[step_idx, agent_idx]),
                "topo_gain": float(topo_gain_hist[step_idx, agent_idx]),
                "psi_tilde": float(psi_hist[step_idx, agent_idx]),
                "phi": float(phi_hist[step_idx, agent_idx]),
                "xi_axis_x": float(xi_axis_hist[step_idx, agent_idx, 0]),
                "xi_axis_y": float(xi_axis_hist[step_idx, agent_idx, 1]),
                "env_axis_x": float(env_axis_hist[step_idx, agent_idx, 0]),
                "env_axis_y": float(env_axis_hist[step_idx, agent_idx, 1]),
                "form_axis_x": float(form_axis_hist[step_idx, agent_idx, 0]),
                "form_axis_y": float(form_axis_hist[step_idx, agent_idx, 1]),
                "tensor_axis_x": float(tensor_axis_hist[step_idx, agent_idx, 0]),
                "tensor_axis_y": float(tensor_axis_hist[step_idx, agent_idx, 1]),
                "env_pressure": float(env_pressure_hist[step_idx, agent_idx]),
                "env_anisotropy": float(env_anisotropy_hist[step_idx, agent_idx]),
                "form_anisotropy": float(form_anisotropy_hist[step_idx, agent_idx]),
                "coverage_aux": float(coverage_aux_hist[step_idx, agent_idx]),
                "nav_scale": float(nav_scale_hist[step_idx, agent_idx]),
                "safe_scale": float(safe_scale_hist[step_idx, agent_idx]),
                "directional_scale": float(directional_scale_hist[step_idx, agent_idx]),
                "guard_scale": float(guard_scale_hist[step_idx, agent_idx]),
                "enc_scale": float(enc_scale_hist[step_idx, agent_idx]),
                "angle_scale": float(angle_scale_hist[step_idx, agent_idx]),
                "magnetic_drive_intent": float(magnetic_drive_intent_hist[step_idx, agent_idx]) if magnetic_drive_intent_hist is not None else float("nan"),
                "magnetic_release_intent": float(magnetic_release_intent_hist[step_idx, agent_idx]) if magnetic_release_intent_hist is not None else float("nan"),
                "magnetic_queue_scale": float(magnetic_queue_scale_hist[step_idx, agent_idx]) if magnetic_queue_scale_hist is not None else float("nan"),
                "magnetic_effective_drive": float(magnetic_effective_drive_hist[step_idx, agent_idx]) if magnetic_effective_drive_hist is not None else float("nan"),
                "magnetic_transport_release_proxy": float(magnetic_transport_release_proxy_hist[step_idx, agent_idx]) if magnetic_transport_release_proxy_hist is not None else float("nan"),
            }
            for z_idx, label in enumerate(DECISION_COLUMNS):
                row[label] = float(decision_z_hist[step_idx, agent_idx, z_idx])
            for x_idx, label in enumerate(XI_COLUMNS):
                row[f"xi_{label}"] = float(xi_state_hist[step_idx, agent_idx, x_idx])
            for mode_idx, label in enumerate(MODE_COLUMNS):
                row[f"mode_logit_{label}"] = float(mode_logits_hist[step_idx, agent_idx, mode_idx])
                row[f"mode_weight_{label}"] = float(mode_weights_hist[step_idx, agent_idx, mode_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def _global_understanding_dataframe(result: dict, method_key: str) -> pd.DataFrame:
    time = _result_time_vector(result)
    positions_hist = result["positions_hist"]
    xi_axis_hist = result["xi_axis_hist"]
    decision_z_hist = result["decision_z_hist"]
    nav_scale_hist = result["nav_scale_hist"]
    safe_scale_hist = result["safe_scale_hist"]
    directional_scale_hist = result["directional_scale_hist"]
    guard_scale_hist = result["guard_scale_hist"]
    enc_scale_hist = result["enc_scale_hist"]
    angle_scale_hist = result["angle_scale_hist"]
    env_pressure_hist = result["env_pressure_hist"]
    env_anisotropy_hist = result["env_anisotropy_hist"]
    form_anisotropy_hist = result["form_anisotropy_hist"]
    tensor_axis_hist = result["tensor_axis_hist"]
    mode_weights_hist = result["mode_weights_hist"]
    directional_axis_hist = result["directional_axis_hist"]
    corridor_activation_hist = result["corridor_activation_hist"]
    queue_activation_hist = result["queue_activation_hist"]
    queue_mode_hist = result["queue_mode_hist"]
    structure_lateral_width_hist = result["structure_lateral_width_hist"]
    structure_longitudinal_span_hist = result["structure_longitudinal_span_hist"]
    global_mode_logits_hist = result.get("global_mode_logits_hist")
    global_mode_weights_hist = result.get("global_mode_weights_hist")
    corridor_confidence_hist = result.get("corridor_confidence_hist")
    corridor_commitment_hist = result.get("corridor_commitment_hist")
    release_need_hist = result.get("release_need_hist")
    xi_feedback_crowding_mean_hist = result.get("xi_feedback_crowding_mean_hist")
    xi_feedback_crowding_peak_hist = result.get("xi_feedback_crowding_peak_hist")
    xi_feedback_slot_order_readiness_hist = result.get("xi_feedback_slot_order_readiness_hist")
    xi_feedback_lateral_order_readiness_hist = result.get("xi_feedback_lateral_order_readiness_hist")
    xi_feedback_forward_room_mean_hist = result.get("xi_feedback_forward_room_mean_hist")
    xi_feedback_obstacle_block_mean_hist = result.get("xi_feedback_obstacle_block_mean_hist")
    xi_feedback_release_ready_hist = result.get("xi_feedback_release_ready_hist")
    guard_release_signal_hist = result.get("guard_release_signal_hist")
    queue_guard_decay_hist = result.get("queue_guard_decay_hist")
    directional_release_signal_hist = result.get("directional_release_signal_hist")
    passage_push_strength_hist = result.get("passage_push_strength_hist")
    entry_progress_hist = result.get("entry_progress_hist")
    traverse_progress_hist = result.get("traverse_progress_hist")
    exit_progress_hist = result.get("exit_progress_hist")
    forward_free_length_hist = result.get("forward_free_length_hist")
    obstacle_pressure_hist = result.get("obstacle_pressure_hist")
    teammate_pressure_hist = result.get("teammate_pressure_hist")
    post_bottleneck_opening_hist = result.get("post_bottleneck_opening_hist")
    front_blocker_bias_hist = result.get("front_blocker_bias_hist")
    magnetic_drive_intent_hist = result.get("magnetic_drive_intent_hist")
    magnetic_release_intent_hist = result.get("magnetic_release_intent_hist")
    magnetic_queue_scale_hist = result.get("magnetic_queue_scale_hist")
    magnetic_effective_drive_hist = result.get("magnetic_effective_drive_hist")
    magnetic_transport_release_proxy_hist = result.get("magnetic_transport_release_proxy_hist")
    spring_chain_storage_hist = result.get("spring_chain_storage_hist")
    spring_head_release_hist = result.get("spring_head_release_hist")
    spring_rear_support_hist = result.get("spring_rear_support_hist")
    spring_link_compression_hist = result.get("spring_link_compression_hist")
    unified_r_hist = result.get("unified_r_hist")
    unified_w_hist = result.get("unified_w_hist")
    unified_q_hist = result.get("unified_q_hist")
    unified_a_hist = result.get("unified_a_hist")
    unified_e_hist = result.get("unified_e_hist")
    unified_entry_push_hist = result.get("unified_entry_push_hist")

    dt = result["params"].dt
    center_hist = np.mean(positions_hist, axis=1)
    center_delta = np.zeros_like(center_hist)
    if len(center_hist) > 1:
        center_delta[1:] = (center_hist[1:] - center_hist[:-1]) / dt
    front_agent_x = np.max(positions_hist[:, :, 0], axis=1)
    rear_agent_x = np.min(positions_hist[:, :, 0], axis=1)
    exit_x = result["scene"].corridor_exit_x
    if exit_x is None:
        agents_past_exit_ratio = np.zeros(len(time), dtype=float)
    else:
        agents_past_exit_ratio = np.mean(positions_hist[:, :, 0] > exit_x, axis=1)

    entry_flux_hist = _positive_derivative(unified_r_hist, dt) if unified_r_hist is not None else np.full(len(time), np.nan)
    if np.ndim(entry_flux_hist) > 0:
        flux_cap = 2.0 * result["params"].u_max
        finite_flux = np.isfinite(entry_flux_hist)
        entry_flux_hist[finite_flux] = np.clip(entry_flux_hist[finite_flux], 0.0, flux_cap)

    rows: list[dict] = []
    for step_idx, t in enumerate(time):
        directional_axis = directional_axis_hist[step_idx]
        forward_speed = float(center_delta[step_idx] @ directional_axis) if np.all(np.isfinite(directional_axis)) else float("nan")
        perp_axis = np.array([-directional_axis[1], directional_axis[0]], dtype=float) if np.all(np.isfinite(directional_axis)) else np.array([np.nan, np.nan], dtype=float)
        effective_lane_count = float("nan")
        if np.all(np.isfinite(perp_axis)):
            lane_ref = max(0.9 * result["params"].r0, result["params"].geo_queue_spacing, 1e-9)
            lateral_coords = (positions_hist[step_idx] - center_hist[step_idx][None, :]) @ perp_axis
            lateral_width = float(np.max(lateral_coords) - np.min(lateral_coords)) if len(lateral_coords) else 0.0
            effective_lane_count = float(np.clip(1.0 + lateral_width / lane_ref, 1.0, positions_hist.shape[1]))
        crowding_peak = float(xi_feedback_crowding_peak_hist[step_idx]) if xi_feedback_crowding_peak_hist is not None else float("nan")
        obstacle_pressure = float(obstacle_pressure_hist[step_idx]) if obstacle_pressure_hist is not None else float("nan")
        teammate_pressure = float(teammate_pressure_hist[step_idx]) if teammate_pressure_hist is not None else float("nan")
        risk_load = float("nan")
        if np.all(np.isfinite([crowding_peak, obstacle_pressure, teammate_pressure])):
            # A compact dissipation proxy: crowding dominates, obstacle/team pressure refine the load.
            risk_load = 0.5 * crowding_peak + 0.25 * obstacle_pressure + 0.25 * teammate_pressure
        if global_mode_weights_hist is not None and not np.all(np.isnan(global_mode_weights_hist[step_idx])):
            mode_weight_mean = np.asarray(global_mode_weights_hist[step_idx], dtype=float)
        else:
            mode_weight_mean = np.array(
                [_nanmean(mode_weights_hist[step_idx, :, mode_idx]) for mode_idx in range(len(MODE_COLUMNS))],
                dtype=float,
            )
        row = {
            "scene": result["scene"].key,
            "method": method_key,
            "seed": result["seed"],
            "step": step_idx,
            "time": float(t),
            "center_x": float(center_hist[step_idx, 0]),
            "center_y": float(center_hist[step_idx, 1]),
            "front_agent_x": float(front_agent_x[step_idx]),
            "rear_agent_x": float(rear_agent_x[step_idx]),
            "agents_past_exit_ratio": float(agents_past_exit_ratio[step_idx]),
            "directional_axis_x": float(directional_axis[0]),
            "directional_axis_y": float(directional_axis[1]),
            "corridor_activation": float(corridor_activation_hist[step_idx]),
            "queue_activation": float(queue_activation_hist[step_idx]),
            "queue_mode": float(queue_mode_hist[step_idx]),
            "corridor_confidence": float(corridor_confidence_hist[step_idx]) if corridor_confidence_hist is not None else float("nan"),
            "corridor_commitment": float(corridor_commitment_hist[step_idx]) if corridor_commitment_hist is not None else float("nan"),
            "release_need": float(release_need_hist[step_idx]) if release_need_hist is not None else float("nan"),
            "xi_feedback_crowding_mean": float(xi_feedback_crowding_mean_hist[step_idx]) if xi_feedback_crowding_mean_hist is not None else float("nan"),
            "xi_feedback_crowding_peak": float(xi_feedback_crowding_peak_hist[step_idx]) if xi_feedback_crowding_peak_hist is not None else float("nan"),
            "xi_feedback_slot_order_readiness": float(xi_feedback_slot_order_readiness_hist[step_idx]) if xi_feedback_slot_order_readiness_hist is not None else float("nan"),
            "xi_feedback_lateral_order_readiness": float(xi_feedback_lateral_order_readiness_hist[step_idx]) if xi_feedback_lateral_order_readiness_hist is not None else float("nan"),
            "xi_feedback_forward_room_mean": float(xi_feedback_forward_room_mean_hist[step_idx]) if xi_feedback_forward_room_mean_hist is not None else float("nan"),
            "xi_feedback_obstacle_block_mean": float(xi_feedback_obstacle_block_mean_hist[step_idx]) if xi_feedback_obstacle_block_mean_hist is not None else float("nan"),
            "xi_feedback_release_ready": float(xi_feedback_release_ready_hist[step_idx]) if xi_feedback_release_ready_hist is not None else float("nan"),
            "guard_release_signal": float(guard_release_signal_hist[step_idx]) if guard_release_signal_hist is not None else float("nan"),
            "queue_guard_decay": float(queue_guard_decay_hist[step_idx]) if queue_guard_decay_hist is not None else float("nan"),
            "directional_release_signal": float(directional_release_signal_hist[step_idx]) if directional_release_signal_hist is not None else float("nan"),
            "passage_push_strength": float(passage_push_strength_hist[step_idx]) if passage_push_strength_hist is not None else float("nan"),
            "entry_progress": float(entry_progress_hist[step_idx]) if entry_progress_hist is not None else float("nan"),
            "traverse_progress": float(traverse_progress_hist[step_idx]) if traverse_progress_hist is not None else float("nan"),
            "exit_progress": float(exit_progress_hist[step_idx]) if exit_progress_hist is not None else float("nan"),
            "forward_free_length": float(forward_free_length_hist[step_idx]) if forward_free_length_hist is not None else float("nan"),
            "obstacle_pressure": obstacle_pressure,
            "teammate_pressure": teammate_pressure,
            "post_bottleneck_opening": float(post_bottleneck_opening_hist[step_idx]) if post_bottleneck_opening_hist is not None else float("nan"),
            "front_blocker_bias": float(front_blocker_bias_hist[step_idx]) if front_blocker_bias_hist is not None else float("nan"),
            "lateral_width": float(structure_lateral_width_hist[step_idx]),
            "longitudinal_span": float(structure_longitudinal_span_hist[step_idx]),
            "effective_lane_count": effective_lane_count,
            "env_pressure_mean": _nanmean(env_pressure_hist[step_idx]),
            "env_pressure_peak": _nanmax(env_pressure_hist[step_idx]),
            "env_anisotropy_mean": _nanmean(env_anisotropy_hist[step_idx]),
            "form_anisotropy_mean": _nanmean(form_anisotropy_hist[step_idx]),
            "progress_mean": _nanmean(decision_z_hist[step_idx, :, 0]),
            "caution_mean": _nanmean(decision_z_hist[step_idx, :, 1]),
            "reshape_mean": _nanmean(decision_z_hist[step_idx, :, 2]),
            "spacing_mean": _nanmean(decision_z_hist[step_idx, :, 3]),
            "nav_scale_mean": _nanmean(nav_scale_hist[step_idx]),
            "safe_scale_mean": _nanmean(safe_scale_hist[step_idx]),
            "directional_scale_mean": _nanmean(directional_scale_hist[step_idx]),
            "guard_scale_mean": _nanmean(guard_scale_hist[step_idx]),
            "enc_scale_mean": _nanmean(enc_scale_hist[step_idx]),
            "angle_scale_mean": _nanmean(angle_scale_hist[step_idx]),
            "tensor_axis_alignment_mean": _alignment_mean(tensor_axis_hist[step_idx], directional_axis),
            "xi_axis_alignment_mean": _alignment_mean(xi_axis_hist[step_idx], directional_axis),
            "forward_speed_along_axis": forward_speed,
            "magnetic_drive_intent_mean": _nanmean(magnetic_drive_intent_hist[step_idx]) if magnetic_drive_intent_hist is not None else float("nan"),
            "magnetic_release_intent_mean": _nanmean(magnetic_release_intent_hist[step_idx]) if magnetic_release_intent_hist is not None else float("nan"),
            "magnetic_queue_scale_mean": _nanmean(magnetic_queue_scale_hist[step_idx]) if magnetic_queue_scale_hist is not None else float("nan"),
            "magnetic_effective_drive_mean": _nanmean(magnetic_effective_drive_hist[step_idx]) if magnetic_effective_drive_hist is not None else float("nan"),
            "magnetic_transport_release_proxy_mean": _nanmean(magnetic_transport_release_proxy_hist[step_idx]) if magnetic_transport_release_proxy_hist is not None else float("nan"),
            "spring_chain_storage": float(spring_chain_storage_hist[step_idx]) if spring_chain_storage_hist is not None else float("nan"),
            "spring_head_release": float(spring_head_release_hist[step_idx]) if spring_head_release_hist is not None else float("nan"),
            "spring_rear_support": float(spring_rear_support_hist[step_idx]) if spring_rear_support_hist is not None else float("nan"),
            "spring_link_compression": float(spring_link_compression_hist[step_idx]) if spring_link_compression_hist is not None else float("nan"),
            "entry_flux": float(entry_flux_hist[step_idx]) if np.ndim(entry_flux_hist) > 0 else float("nan"),
            "risk_load": risk_load,
            "unified_r": float(unified_r_hist[step_idx]) if unified_r_hist is not None else float("nan"),
            "unified_w": float(unified_w_hist[step_idx]) if unified_w_hist is not None else float("nan"),
            "unified_q": float(unified_q_hist[step_idx]) if unified_q_hist is not None else float("nan"),
            "unified_a": float(unified_a_hist[step_idx]) if unified_a_hist is not None else float("nan"),
            "unified_e": float(unified_e_hist[step_idx]) if unified_e_hist is not None else float("nan"),
            "unified_entry_push": float(unified_entry_push_hist[step_idx]) if unified_entry_push_hist is not None else float("nan"),
            "mode_entropy": _mode_entropy(mode_weight_mean),
            "dominant_mode": _dominant_mode(mode_weight_mean),
        }
        for mode_idx, label in enumerate(MODE_COLUMNS):
            row[f"mode_weight_{label}_mean"] = float(mode_weight_mean[mode_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _understanding_trial_summary(result: dict, method_key: str, global_df: pd.DataFrame) -> dict:
    metrics = result["metrics"]
    scene = result["scene"]
    late_mask = _late_mask(len(global_df))
    post_exit_mask = _corridor_post_exit_mask(scene, global_df["agents_past_exit_ratio"].to_numpy())
    dominant_mode_switches = _dominant_mode_switch_count(global_df["dominant_mode"].tolist())

    summary = {
        "scene": scene.key,
        "method": method_key,
        "seed": result["seed"],
        "success": float(metrics.success),
        "collisions": float(metrics.collisions),
        "no_collision": float(metrics.no_collision),
        "completion_time": float(metrics.completion_time),
        "min_obs_distance": float(metrics.min_obs_distance),
        "min_agent_distance": float(metrics.min_agent_distance),
        "queue_stability": float(metrics.queue_stability),
        "corridor_activation": _nanmean(global_df["corridor_activation"].to_numpy()),
        "queue_activation": _nanmean(global_df["queue_activation"].to_numpy()),
        "queue_mode": _nanmean(global_df["queue_mode"].to_numpy()),
        "env_pressure": _nanmean(global_df["env_pressure_mean"].to_numpy()),
        "env_pressure_peak": _nanmax(global_df["env_pressure_peak"].to_numpy()),
        "env_anisotropy": _nanmean(global_df["env_anisotropy_mean"].to_numpy()),
        "form_anisotropy": _nanmean(global_df["form_anisotropy_mean"].to_numpy()),
        "tensor_axis_alignment": _nanmean(global_df["tensor_axis_alignment_mean"].to_numpy()),
        "xi_axis_alignment": _nanmean(global_df["xi_axis_alignment_mean"].to_numpy()),
        "mode_entropy": _nanmean(global_df["mode_entropy"].to_numpy()),
        "dominant_mode_switches": float(dominant_mode_switches),
        "progress": _nanmean(global_df["progress_mean"].to_numpy()),
        "caution": _nanmean(global_df["caution_mean"].to_numpy()),
        "reshape": _nanmean(global_df["reshape_mean"].to_numpy()),
        "spacing": _nanmean(global_df["spacing_mean"].to_numpy()),
        "xi_feedback_crowding_mean": _nanmean(global_df["xi_feedback_crowding_mean"].to_numpy()),
        "xi_feedback_crowding_peak": _nanmean(global_df["xi_feedback_crowding_peak"].to_numpy()),
        "xi_feedback_slot_order_readiness": _nanmean(global_df["xi_feedback_slot_order_readiness"].to_numpy()),
        "xi_feedback_lateral_order_readiness": _nanmean(global_df["xi_feedback_lateral_order_readiness"].to_numpy()),
        "xi_feedback_forward_room_mean": _nanmean(global_df["xi_feedback_forward_room_mean"].to_numpy()),
        "xi_feedback_obstacle_block_mean": _nanmean(global_df["xi_feedback_obstacle_block_mean"].to_numpy()),
        "xi_feedback_release_ready": _nanmean(global_df["xi_feedback_release_ready"].to_numpy()),
        "guard_release_signal": _nanmean(global_df["guard_release_signal"].to_numpy()),
        "queue_guard_decay": _nanmean(global_df["queue_guard_decay"].to_numpy()),
        "directional_release_signal": _nanmean(global_df["directional_release_signal"].to_numpy()),
        "passage_push_strength": _nanmean(global_df["passage_push_strength"].to_numpy()),
        "nav_scale": _nanmean(global_df["nav_scale_mean"].to_numpy()),
        "safe_scale": _nanmean(global_df["safe_scale_mean"].to_numpy()),
        "directional_scale": _nanmean(global_df["directional_scale_mean"].to_numpy()),
        "guard_scale": _nanmean(global_df["guard_scale_mean"].to_numpy()),
        "forward_speed_along_axis": _nanmean(global_df["forward_speed_along_axis"].to_numpy()),
        "magnetic_drive_intent": _nanmean(global_df["magnetic_drive_intent_mean"].to_numpy()),
        "magnetic_release_intent": _nanmean(global_df["magnetic_release_intent_mean"].to_numpy()),
        "magnetic_queue_scale": _nanmean(global_df["magnetic_queue_scale_mean"].to_numpy()),
        "magnetic_effective_drive": _nanmean(global_df["magnetic_effective_drive_mean"].to_numpy()),
        "magnetic_transport_release_proxy": _nanmean(global_df["magnetic_transport_release_proxy_mean"].to_numpy()),
        "spring_chain_storage": _nanmean(global_df["spring_chain_storage"].to_numpy()),
        "spring_head_release": _nanmean(global_df["spring_head_release"].to_numpy()),
        "spring_rear_support": _nanmean(global_df["spring_rear_support"].to_numpy()),
        "spring_link_compression": _nanmean(global_df["spring_link_compression"].to_numpy()),
        "effective_lane_count": _nanmean(global_df["effective_lane_count"].to_numpy()),
        "entry_flux": _nanmean(global_df["entry_flux"].to_numpy()),
        "entry_flux_peak": _nanmax(global_df["entry_flux"].to_numpy()),
        "risk_load": _nanmean(global_df["risk_load"].to_numpy()),
        "risk_load_peak": _nanmax(global_df["risk_load"].to_numpy()),
        "unified_r_peak": _nanmax(global_df["unified_r"].to_numpy()),
        "unified_r_late": _nanmean(global_df.loc[late_mask, "unified_r"].to_numpy()),
        "unified_e": _nanmean(global_df["unified_e"].to_numpy()),
        "unified_e_peak": _nanmax(global_df["unified_e"].to_numpy()),
        "late_progress": _nanmean(global_df.loc[late_mask, "progress_mean"].to_numpy()),
        "late_caution": _nanmean(global_df.loc[late_mask, "caution_mean"].to_numpy()),
        "late_nav_scale": _nanmean(global_df.loc[late_mask, "nav_scale_mean"].to_numpy()),
        "late_guard_scale": _nanmean(global_df.loc[late_mask, "guard_scale_mean"].to_numpy()),
        "late_forward_speed_along_axis": _nanmean(global_df.loc[late_mask, "forward_speed_along_axis"].to_numpy()),
        "late_magnetic_drive_intent": _nanmean(global_df.loc[late_mask, "magnetic_drive_intent_mean"].to_numpy()),
        "late_magnetic_release_intent": _nanmean(global_df.loc[late_mask, "magnetic_release_intent_mean"].to_numpy()),
        "late_magnetic_queue_scale": _nanmean(global_df.loc[late_mask, "magnetic_queue_scale_mean"].to_numpy()),
        "late_magnetic_effective_drive": _nanmean(global_df.loc[late_mask, "magnetic_effective_drive_mean"].to_numpy()),
        "late_magnetic_transport_release_proxy": _nanmean(global_df.loc[late_mask, "magnetic_transport_release_proxy_mean"].to_numpy()),
        "late_spring_chain_storage": _nanmean(global_df.loc[late_mask, "spring_chain_storage"].to_numpy()),
        "late_spring_head_release": _nanmean(global_df.loc[late_mask, "spring_head_release"].to_numpy()),
        "late_spring_rear_support": _nanmean(global_df.loc[late_mask, "spring_rear_support"].to_numpy()),
        "late_effective_lane_count": _nanmean(global_df.loc[late_mask, "effective_lane_count"].to_numpy()),
        "late_entry_flux": _nanmean(global_df.loc[late_mask, "entry_flux"].to_numpy()),
        "late_risk_load": _nanmean(global_df.loc[late_mask, "risk_load"].to_numpy()),
        "late_xi_feedback_release_ready": _nanmean(global_df.loc[late_mask, "xi_feedback_release_ready"].to_numpy()),
        "late_guard_release_signal": _nanmean(global_df.loc[late_mask, "guard_release_signal"].to_numpy()),
        "late_queue_guard_decay": _nanmean(global_df.loc[late_mask, "queue_guard_decay"].to_numpy()),
        "late_directional_release_signal": _nanmean(global_df.loc[late_mask, "directional_release_signal"].to_numpy()),
        "late_passage_push_strength": _nanmean(global_df.loc[late_mask, "passage_push_strength"].to_numpy()),
    }
    for label in MODE_COLUMNS:
        summary[f"mode_weight_{label}"] = _nanmean(global_df[f"mode_weight_{label}_mean"].to_numpy())

    if scene.corridor_exit_x is not None and np.any(post_exit_mask):
        summary.update(
            {
                "post_exit_available": 1.0,
                "post_exit_steps": float(np.sum(post_exit_mask)),
                "post_exit_agents_past_exit_ratio": _nanmean(global_df.loc[post_exit_mask, "agents_past_exit_ratio"].to_numpy()),
                "post_exit_progress": _nanmean(global_df.loc[post_exit_mask, "progress_mean"].to_numpy()),
                "post_exit_caution": _nanmean(global_df.loc[post_exit_mask, "caution_mean"].to_numpy()),
                "post_exit_nav_scale": _nanmean(global_df.loc[post_exit_mask, "nav_scale_mean"].to_numpy()),
                "post_exit_guard_scale": _nanmean(global_df.loc[post_exit_mask, "guard_scale_mean"].to_numpy()),
                "post_exit_forward_speed_along_axis": _nanmean(global_df.loc[post_exit_mask, "forward_speed_along_axis"].to_numpy()),
                "post_exit_guard_over_nav_rate": float(np.mean((global_df.loc[post_exit_mask, "guard_scale_mean"] > global_df.loc[post_exit_mask, "nav_scale_mean"]).to_numpy())),
                "post_exit_caution_over_progress_rate": float(np.mean((global_df.loc[post_exit_mask, "caution_mean"] > global_df.loc[post_exit_mask, "progress_mean"]).to_numpy())),
            }
        )
        for label in MODE_COLUMNS:
            summary[f"post_exit_mode_weight_{label}"] = _nanmean(global_df.loc[post_exit_mask, f"mode_weight_{label}_mean"].to_numpy())
    else:
        summary.update(
            {
                "post_exit_available": 0.0,
                "post_exit_steps": 0.0,
                "post_exit_agents_past_exit_ratio": np.nan,
                "post_exit_progress": np.nan,
                "post_exit_caution": np.nan,
                "post_exit_nav_scale": np.nan,
                "post_exit_guard_scale": np.nan,
                "post_exit_forward_speed_along_axis": np.nan,
                "post_exit_guard_over_nav_rate": np.nan,
                "post_exit_caution_over_progress_rate": np.nan,
            }
        )
        for label in MODE_COLUMNS:
            summary[f"post_exit_mode_weight_{label}"] = np.nan
    return summary


def _write_understanding_bundle(result: dict, method_key: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_df = _agent_understanding_dataframe(result, method_key)
    global_df = _global_understanding_dataframe(result, method_key)
    agent_df.to_csv(output_dir / "agent_understanding.csv", index=False)
    global_df.to_csv(output_dir / "global_understanding.csv", index=False)

    summary = _understanding_trial_summary(result, method_key, global_df)
    with (output_dir / "understanding_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(result["metrics"]), fh, ensure_ascii=False, indent=2)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "scene": result["scene"].key,
                "method": method_key,
                "method_name": result["method"].name,
                "seed": result["seed"],
                "params": asdict(result["params"]),
            },
            fh,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return summary


def export_understanding_trial(scene_key: str, method_key: str, seed: int, output_dir: Path, t_pred: float | None = None) -> dict:
    scene = SCENARIOS[scene_key]
    params = replace(scene.params, t_pred=t_pred) if t_pred is not None else scene.params
    result = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed, params_override=params)
    summary = _write_understanding_bundle(result, method_key, output_dir)
    return {
        "scene": scene_key,
        "method": method_key,
        "seed": seed,
        "output": str(output_dir),
        "summary": summary,
    }


def run_understanding_analysis(
    output_dir: str | Path,
    scenes: list[str] | None = None,
    methods: list[str] | None = None,
    trials: int = 5,
    seed_start: int = 0,
    t_pred: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_scenes = scenes or [
        "c_geo_directional_passage_lite",
        "d_geo_fast_target_lite",
        "e_geo_tracking_single_obstacle_lite",
    ]
    selected_methods = methods or ["M21", "M22", "M23", "M24"]
    if methods is None:
        selected_methods.append("M28")
        selected_methods.append("M29")
        selected_methods.append("M30")

    rows: list[dict] = []
    for scene_key in selected_scenes:
        for method_key in selected_methods:
            for seed in range(seed_start, seed_start + trials):
                scene = SCENARIOS[scene_key]
                params = replace(scene.params, t_pred=t_pred) if t_pred is not None else scene.params
                result = run_scene_trial(scene, METHOD_LIBRARY[method_key], seed=seed, params_override=params)
                trial_output = output_path / "trials" / scene_key / method_key / f"seed_{seed}"
                rows.append(_write_understanding_bundle(result, method_key, trial_output))

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(output_path / "understanding_trial_summary.csv", index=False)
    summary_df = detail_df.groupby(["scene", "method"]).mean(numeric_only=True).reset_index()
    summary_df = summary_df.rename(
        columns={
            column: (
                f"{column}_rate" if column in {"success", "collisions", "no_collision", "post_exit_available"} else f"{column}_mean"
            )
            for column in summary_df.columns
            if column not in {"scene", "method"}
        }
    )
    summary_df.to_csv(output_path / "understanding_summary.csv", index=False)
    with (output_path / "understanding_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_df.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)
    with (output_path / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "scenes": selected_scenes,
                "methods": selected_methods,
                "trials": trials,
                "seed_start": seed_start,
                "t_pred": t_pred,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    return detail_df, summary_df
