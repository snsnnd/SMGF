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
    inside_final: bool
    radius_error_final: float
    success_geom_final: bool
    success_no_collision: bool
    gmax_reach_time: float


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
    final_positions = positions_hist[-1]
    final_target = target_hist[-1]
    final_predicted = predicted_target_hist[-1]
    mean_radius, sigma_r2 = radius_stats(final_positions, final_target)
    gmax_deg = np.degrees(max_angle_gap(final_positions, final_target)) if len(final_positions) > 1 else 0.0
    centroid = np.mean(final_positions, axis=0)
    hull = monotonic_chain(final_positions)
    inside = point_in_convex_polygon(final_target, hull)
    current_center_error = norm(centroid - final_target)
    predicted_center_error = norm(centroid - final_predicted)
    min_obs = min(min_obstacle_distance(step, eval_obstacles) for step in positions_hist)
    min_agent = min(min_agent_distance(step) for step in positions_hist)
    collisions = min_obs < params.d_obs_safe or min_agent < params.d_agent_safe
    radius_error_final = abs(mean_radius - params.r_c)
    success_geom_final = (
        inside
        and gmax_deg <= params.gmax_threshold_deg
        and radius_error_final <= params.radius_tolerance
        and sigma_r2 <= params.sigma_r_threshold
    )
    success_no_collision = not collisions
    gmax_reach_time = params.horizon
    if len(final_positions) > 1:
        for k in range(len(positions_hist)):
            gmax_k = np.degrees(max_angle_gap(positions_hist[k], target_hist[k]))
            if gmax_k <= 120.0:
                gmax_reach_time = float(k * dt)
                break
    control_cost = float(np.sum(np.linalg.norm(u_hist, axis=2) ** 2) * dt)
    control_delta = np.diff(u_hist, axis=0)
    control_smoothness = float(np.sum(np.linalg.norm(control_delta, axis=2) ** 2)) if len(control_delta) else 0.0
    sat_ratio = float(np.mean(np.linalg.norm(u_hist, axis=2) >= 0.95 * params.u_max))
    path_length = float(np.sum(np.linalg.norm(np.diff(positions_hist, axis=0), axis=2))) if len(positions_hist) > 1 else 0.0
    speed_hist = np.linalg.norm(u_hist, axis=2)
    stall_steps = int(np.sum(np.mean(speed_hist, axis=1) < 0.05))
    angles = np.arctan2(final_positions[:, 1] - final_target[1], final_positions[:, 0] - final_target[0]) if len(final_positions) else np.array([0.0])
    convoy_ratio = float(np.ptp(angles) < np.pi)

    success = False
    completion_time = params.horizon
    if scenario.success_mode == "goal_reach":
        success_steps = np.linalg.norm(positions_hist[:, 0, :] - target_hist, axis=1) < 0.8
        if np.any(success_steps) and not collisions:
            completion_time = float(np.argmax(success_steps) * dt)
            success = True
    elif scenario.success_mode == "encirclement":
        dwell_steps = max(1, int(1.0 / dt))
        streak = 0
        for k in range(len(positions_hist)):
            step_positions = positions_hist[k]
            step_target = target_hist[k]
            hull_k = monotonic_chain(step_positions)
            inside_k = point_in_convex_polygon(step_target, hull_k)
            gmax_k = np.degrees(max_angle_gap(step_positions, step_target))
            mean_r_k, sigma_r_k = radius_stats(step_positions, step_target)
            if inside_k and gmax_k <= params.gmax_threshold_deg and abs(mean_r_k - params.r_c) <= params.radius_tolerance and sigma_r_k <= params.sigma_r_threshold:
                streak += 1
                if streak >= dwell_steps:
                    completion_time = float((k - dwell_steps + 1) * dt)
                    success = not collisions
                    break
            else:
                streak = 0
    elif scenario.success_mode == "corridor_pass":
        threshold = scenario.corridor_exit_x if scenario.corridor_exit_x is not None else 0.0
        dwell_steps = max(1, int(0.5 / dt))
        streak = 0
        for k in range(len(positions_hist)):
            if np.all(positions_hist[k, :, 0] > threshold):
                streak += 1
                if streak >= dwell_steps:
                    completion_time = float((k - dwell_steps + 1) * dt)
                    success = not collisions
                    break
            else:
                streak = 0
    elif scenario.success_mode == "target_track":
        distances = np.mean(np.linalg.norm(positions_hist - target_hist[:, None, :], axis=2), axis=1)
        mask = distances < params.r_c
        if np.any(mask) and not collisions:
            completion_time = float(np.argmax(mask) * dt)
            success = True

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
        inside_final=inside,
        radius_error_final=radius_error_final,
        success_geom_final=success_geom_final,
        success_no_collision=success_no_collision,
        gmax_reach_time=gmax_reach_time,
    )
