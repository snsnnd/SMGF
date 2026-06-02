from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import EPS, Obstacle, Params, Scenario, Target, norm


def monotonic_chain(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points, axis=0)
    if len(pts) <= 1:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    sign = None
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) <= 1e-9:
            continue
        current = cross > 0
        if sign is None:
            sign = current
        elif sign != current:
            return False
    return True


def max_angle_gap(positions: np.ndarray, center: np.ndarray) -> float:
    angles = np.sort(np.mod(np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0]), 2 * np.pi))
    wrapped = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    return float(np.max(np.diff(wrapped)))


def radius_stats(positions: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    radii = np.linalg.norm(positions - center[None, :], axis=1)
    mean_r = float(np.mean(radii))
    return mean_r, float(np.mean((radii - mean_r) ** 2))


def min_obstacle_distance(positions: np.ndarray, obstacles: list[Obstacle]) -> float:
    if not obstacles:
        return float("inf")
    values = []
    for p in positions:
        for obs in obstacles:
            values.append(norm(p - obs.center) - obs.radius)
    return float(min(values)) if values else float("inf")


def min_agent_distance(positions: np.ndarray) -> float:
    if len(positions) <= 1:
        return float("inf")
    minimum = float("inf")
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            minimum = min(minimum, norm(positions[i] - positions[j]))
    return minimum


@dataclass(frozen=True)
class TrialMetrics:
    success: bool
    completion_time: float
    min_obs_distance: float
    min_agent_distance: float
    collisions: bool
    control_cost: float
    control_smoothness: float
    input_saturation_ratio: float
    max_angle_gap_deg: float
    radius_variance: float
    mean_radius: float
    current_center_error: float
    predicted_center_error: float
    path_length: float
    stall_steps: int
    convoy_ratio: float
    inside_any: bool
    inside_final: bool
    gmax_success: bool
    radius_success: bool
    sigma_success: bool
    no_collision: bool
    obs_collision: bool
    agent_collision: bool
    dwell_success: bool
    dwell_geom_success: bool
    dwell_success_no_collision: bool
    radius_error_final: float
    success_geom_final: bool
    success_no_collision: bool
    gmax_reach_time: float
    time_to_inside: float
    time_to_gmax: float
    time_to_radius: float
    time_to_full_geom: float
    success_hold_time: float
    post_success_violation_count: int
    last_10s_gmax_mean: float
    last_10s_radius_error_mean: float
    last_10s_inside_rate: float


def _first_true_time(mask: np.ndarray, dt: float, default: float) -> float:
    if np.any(mask):
        return float(np.argmax(mask) * dt)
    return default


def _dwell_start(mask: np.ndarray, dwell_steps: int) -> int | None:
    streak = 0
    for idx, is_true in enumerate(mask):
        if is_true:
            streak += 1
            if streak >= dwell_steps:
                return idx - dwell_steps + 1
        else:
            streak = 0
    return None


def _hold_stats(mask: np.ndarray, start_idx: int | None, dt: float) -> tuple[float, int]:
    if start_idx is None:
        return 0.0, 0
    hold_steps = 0
    idx = start_idx
    while idx < len(mask) and mask[idx]:
        hold_steps += 1
        idx += 1
    violation_count = 0
    in_violation = False
    for is_true in mask[idx:]:
        if not is_true and not in_violation:
            violation_count += 1
            in_violation = True
        elif is_true:
            in_violation = False
    return float(hold_steps * dt), violation_count


def evaluate_trial(
    scenario: Scenario,
    params: Params,
    positions_hist: np.ndarray,
    target_hist: np.ndarray,
    predicted_target_hist: np.ndarray,
    u_hist: np.ndarray,
    dt: float,
    obstacles: list[Obstacle] | None = None,
) -> TrialMetrics:
    eval_obstacles = scenario.obstacles if obstacles is None else obstacles
    steps = len(positions_hist)
    final_positions = positions_hist[-1]
    final_target = target_hist[-1]
    final_predicted = predicted_target_hist[-1]
    inside_series = np.zeros(steps, dtype=bool)
    gmax_deg_series = np.zeros(steps)
    mean_radius_series = np.zeros(steps)
    sigma_series = np.zeros(steps)
    radius_error_series = np.zeros(steps)
    obs_distance_series = np.zeros(steps)
    agent_distance_series = np.zeros(steps)
    obs_collision_series = np.zeros(steps, dtype=bool)
    agent_collision_series = np.zeros(steps, dtype=bool)
    for k in range(steps):
        step_positions = positions_hist[k]
        step_target = target_hist[k]
        hull_k = monotonic_chain(step_positions)
        inside_series[k] = point_in_convex_polygon(step_target, hull_k)
        gmax_deg_series[k] = np.degrees(max_angle_gap(step_positions, step_target)) if len(step_positions) > 1 else 0.0
        mean_r_k, sigma_r_k = radius_stats(step_positions, step_target)
        mean_radius_series[k] = mean_r_k
        sigma_series[k] = sigma_r_k
        radius_error_series[k] = abs(mean_r_k - params.r_c)
        obs_distance_series[k] = min_obstacle_distance(step_positions, eval_obstacles)
        agent_distance_series[k] = min_agent_distance(step_positions)
        obs_collision_series[k] = obs_distance_series[k] < params.d_obs_safe
        agent_collision_series[k] = agent_distance_series[k] < params.d_agent_safe if params.d_agent_safe > 0 else False

    mean_radius = float(mean_radius_series[-1])
    sigma_r2 = float(sigma_series[-1])
    gmax_deg = float(gmax_deg_series[-1])
    centroid = np.mean(final_positions, axis=0)
    hull = monotonic_chain(final_positions)
    inside = point_in_convex_polygon(final_target, hull)
    inside_any = bool(np.any(inside_series))
    current_center_error = norm(centroid - final_target)
    predicted_center_error = norm(centroid - final_predicted)
    min_obs = float(np.min(obs_distance_series))
    min_agent = float(np.min(agent_distance_series))
    obs_collision = bool(np.any(obs_collision_series))
    agent_collision = bool(np.any(agent_collision_series))
    collisions = obs_collision or agent_collision
    radius_error_final = float(radius_error_series[-1])
    gmax_success = gmax_deg <= params.gmax_threshold_deg
    radius_success = radius_error_final <= params.radius_tolerance
    sigma_success = sigma_r2 <= params.sigma_r_threshold
    success_geom_final = inside and gmax_success and radius_success and sigma_success
    no_collision = not collisions
    success_no_collision = no_collision
    gmax_success_series = gmax_deg_series <= params.gmax_threshold_deg
    radius_success_series = radius_error_series <= params.radius_tolerance
    sigma_success_series = sigma_series <= params.sigma_r_threshold
    geom_success_series = inside_series & gmax_success_series & radius_success_series & sigma_success_series
    time_to_inside = _first_true_time(inside_series, dt, params.horizon)
    time_to_gmax = _first_true_time(gmax_success_series, dt, params.horizon)
    time_to_radius = _first_true_time(radius_success_series, dt, params.horizon)
    time_to_full_geom = _first_true_time(geom_success_series, dt, params.horizon)
    gmax_reach_time = time_to_gmax
    control_cost = float(np.sum(np.linalg.norm(u_hist, axis=2) ** 2) * dt)
    control_delta = np.diff(u_hist, axis=0)
    control_smoothness = float(np.sum(np.linalg.norm(control_delta, axis=2) ** 2)) if len(control_delta) else 0.0
    sat_ratio = float(np.mean(np.linalg.norm(u_hist, axis=2) >= 0.95 * params.u_max))
    path_length = float(np.sum(np.linalg.norm(np.diff(positions_hist, axis=0), axis=2))) if len(positions_hist) > 1 else 0.0
    speed_hist = np.linalg.norm(u_hist, axis=2)
    stall_steps = int(np.sum(np.mean(speed_hist, axis=1) < 0.05))
    angles = np.arctan2(final_positions[:, 1] - final_target[1], final_positions[:, 0] - final_target[0]) if len(final_positions) else np.array([0.0])
    convoy_ratio = float(np.ptp(angles) < np.pi)
    tail_steps = max(1, int(10.0 / dt))
    last_slice = slice(max(0, steps - tail_steps), steps)
    last_10s_gmax_mean = float(np.mean(gmax_deg_series[last_slice]))
    last_10s_radius_error_mean = float(np.mean(radius_error_series[last_slice]))
    last_10s_inside_rate = float(np.mean(inside_series[last_slice]))

    success = False
    completion_time = params.horizon
    dwell_success = False
    dwell_geom_success = False
    dwell_start_idx: int | None = None
    if scenario.success_mode == "goal_reach":
        base_success_series = np.linalg.norm(positions_hist[:, 0, :] - target_hist, axis=1) < 0.8
        dwell_start_idx = _dwell_start(base_success_series, 1)
        dwell_success = dwell_start_idx is not None
        dwell_geom_success = dwell_success
    elif scenario.success_mode == "encirclement":
        dwell_steps = max(1, int(1.0 / dt))
        dwell_start_idx = _dwell_start(geom_success_series, dwell_steps)
        dwell_success = dwell_start_idx is not None
        dwell_geom_success = dwell_success
    elif scenario.success_mode == "corridor_pass":
        threshold = scenario.corridor_exit_x if scenario.corridor_exit_x is not None else 0.0
        dwell_steps = max(1, int(0.5 / dt))
        base_success_series = np.all(positions_hist[:, :, 0] > threshold, axis=1)
        dwell_start_idx = _dwell_start(base_success_series, dwell_steps)
        dwell_success = dwell_start_idx is not None
        dwell_geom_success = dwell_success
    elif scenario.success_mode == "target_track":
        distances = np.mean(np.linalg.norm(positions_hist - target_hist[:, None, :], axis=2), axis=1)
        base_success_series = distances < params.r_c
        dwell_start_idx = _dwell_start(base_success_series, 1)
        dwell_success = dwell_start_idx is not None
        dwell_geom_success = dwell_success

    if dwell_start_idx is not None:
        completion_time = float(dwell_start_idx * dt)
    dwell_success_no_collision = dwell_success and no_collision
    success = dwell_success_no_collision
    if scenario.success_mode == "encirclement":
        hold_mask = geom_success_series
    elif scenario.success_mode == "corridor_pass":
        threshold = scenario.corridor_exit_x if scenario.corridor_exit_x is not None else 0.0
        hold_mask = np.all(positions_hist[:, :, 0] > threshold, axis=1)
    elif scenario.success_mode == "goal_reach":
        hold_mask = np.linalg.norm(positions_hist[:, 0, :] - target_hist, axis=1) < 0.8
    else:
        hold_mask = np.mean(np.linalg.norm(positions_hist - target_hist[:, None, :], axis=2), axis=1) < params.r_c
    success_hold_time, post_success_violation_count = _hold_stats(hold_mask, dwell_start_idx, dt)

    return TrialMetrics(
        success=success,
        completion_time=completion_time,
        min_obs_distance=min_obs,
        min_agent_distance=min_agent,
        collisions=collisions,
        control_cost=control_cost,
        control_smoothness=control_smoothness,
        input_saturation_ratio=sat_ratio,
        max_angle_gap_deg=gmax_deg,
        radius_variance=sigma_r2,
        mean_radius=mean_radius,
        current_center_error=current_center_error,
        predicted_center_error=predicted_center_error,
        path_length=path_length,
        stall_steps=stall_steps,
        convoy_ratio=convoy_ratio,
        inside_any=inside_any,
        inside_final=inside,
        gmax_success=gmax_success,
        radius_success=radius_success,
        sigma_success=sigma_success,
        no_collision=no_collision,
        obs_collision=obs_collision,
        agent_collision=agent_collision,
        dwell_success=dwell_success,
        dwell_geom_success=dwell_geom_success,
        dwell_success_no_collision=dwell_success_no_collision,
        radius_error_final=radius_error_final,
        success_geom_final=success_geom_final,
        success_no_collision=success_no_collision,
        gmax_reach_time=gmax_reach_time,
        time_to_inside=time_to_inside,
        time_to_gmax=time_to_gmax,
        time_to_radius=time_to_radius,
        time_to_full_geom=time_to_full_geom,
        success_hold_time=success_hold_time,
        post_success_violation_count=post_success_violation_count,
        last_10s_gmax_mean=last_10s_gmax_mean,
        last_10s_radius_error_mean=last_10s_radius_error_mean,
        last_10s_inside_rate=last_10s_inside_rate,
    )
