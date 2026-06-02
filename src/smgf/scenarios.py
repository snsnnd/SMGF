from __future__ import annotations

from dataclasses import replace

import numpy as np

from .core import Obstacle, Params, Scenario, Target


def _line_target(position: np.ndarray, velocity: np.ndarray):
    def target_fn(t: float) -> Target:
        return Target(position=position + velocity * t, velocity=velocity.copy())

    return target_fn


def _turning_target(position: np.ndarray, speed: float, omega: float):
    def target_fn(t: float) -> Target:
        vel = speed * np.array([np.cos(omega * t), np.sin(omega * t)])
        pos = position + np.array([speed * np.sin(omega * t) / max(omega, 1e-6), speed * (1.0 - np.cos(omega * t)) / max(omega, 1e-6)])
        return Target(position=pos, velocity=vel)

    return target_fn


def _corridor_obstacles(y_center: float, radius: float, x_positions: tuple[float, float] = (0.0, 4.0)) -> list[Obstacle]:
    return [
        Obstacle(center=np.array([x_positions[0], y_center]), radius=radius),
        Obstacle(center=np.array([x_positions[0], -y_center]), radius=radius),
        Obstacle(center=np.array([x_positions[1], y_center]), radius=radius),
        Obstacle(center=np.array([x_positions[1], -y_center]), radius=radius),
    ]


def build_scenarios() -> dict[str, Scenario]:
    base = Params()

    s1 = Scenario(
        key="s1_single_obstacle",
        title="S1 Single-Agent Obstacle Avoidance",
        description="Single agent, near-collinear obstacle-goal geometry for APF local-minimum verification.",
        n_agents=1,
        obstacles=[Obstacle(center=np.array([0.0, 0.0]), radius=1.1)],
        initial_positions=np.array([[-6.0, 0.25]]),
        target_fn=_line_target(np.array([6.0, 0.0]), np.zeros(2)),
        params=replace(base, horizon=35.0, t_pred=0.0, k_r=0.0, r_c=0.0),
        success_mode="goal_reach",
    )

    s2 = Scenario(
        key="s2_same_side_expansion",
        title="S2 Same-Side Expansion",
        description="Concentrated same-side start that stresses Psi-driven angular expansion.",
        n_agents=6,
        obstacles=[],
        initial_positions=np.array([
            [-5.0, -0.30],
            [-5.2, -0.18],
            [-5.1, -0.06],
            [-5.3, 0.06],
            [-5.2, 0.18],
            [-5.0, 0.30],
        ]),
        target_fn=_line_target(np.array([2.0, 0.0]), np.zeros(2)),
        params=replace(base, horizon=80.0, t_pred=0.0, r_c=2.8, k_t=1.2, r0=1.5, radius_tolerance=0.8, gmax_threshold_deg=140.0),
        success_mode="encirclement",
    )

    s3 = Scenario(
        key="s3_static_encirclement",
        title="S3 Static Encirclement",
        description="Distributed agents contract toward a target ring after initial expansion.",
        n_agents=6,
        obstacles=[],
        initial_positions=np.array([[-4.0, -4.0], [-2.8, 3.7], [-0.5, -5.0], [2.0, 4.4], [4.8, -2.7], [6.0, 3.2]]),
        target_fn=_line_target(np.array([0.0, 0.0]), np.zeros(2)),
        params=replace(base, horizon=50.0, r_c=1.7, radius_tolerance=0.4),
        success_mode="encirclement",
    )

    obstacles_s4_easy = _corridor_obstacles(y_center=2.8, radius=1.35)
    obstacles_s4_medium = _corridor_obstacles(y_center=2.45, radius=1.45)
    obstacles_s4_hard = _corridor_obstacles(y_center=2.3, radius=1.55)
    s4_easy = Scenario(
        key="s4_narrow_passage_easy",
        title="S4 Easy Narrow Passage",
        description="Wide corridor used to verify that pressure modulation can pass an easy bottleneck before harder cases.",
        n_agents=6,
        obstacles=obstacles_s4_easy,
        initial_positions=np.array([[-6.0, -1.8], [-6.0, -1.1], [-6.0, -0.4], [-6.0, 0.3], [-6.0, 1.0], [-6.0, 1.7]]),
        target_fn=_line_target(np.array([12.0, 0.0]), np.zeros(2)),
        params=replace(base, horizon=32.0, k_r=0.0, k_t=0.9, r0=1.5, eta_min=0.05, sigma_omega=8.0, d_agent_safe=0.28, d_obs_safe=0.25, k_s=4.0),
        success_mode="corridor_pass",
        corridor_exit_x=5.2,
    )
    s4_medium = Scenario(
        key="s4_narrow_passage_medium",
        title="S4 Medium Narrow Passage",
        description="Medium corridor retained as the main Omega validation case.",
        n_agents=6,
        obstacles=obstacles_s4_medium,
        initial_positions=np.array([[-6.0, -1.5], [-6.1, -0.9], [-6.0, -0.3], [-6.0, 0.3], [-6.1, 0.9], [-6.0, 1.5]]),
        target_fn=_line_target(np.array([12.0, 0.0]), np.zeros(2)),
        params=replace(base, horizon=40.0, k_r=0.0, k_t=1.2, r0=1.6, eta_min=0.06, sigma_omega=8.0, d_agent_safe=0.34, d_obs_safe=0.28, k_s=3.4),
        success_mode="corridor_pass",
        corridor_exit_x=5.2,
    )
    s4_hard = Scenario(
        key="s4_narrow_passage_hard",
        title="S4 Hard Narrow Passage",
        description="Tight corridor used as a stress test after easier cases are stable.",
        n_agents=6,
        obstacles=obstacles_s4_hard,
        initial_positions=np.array([[-6.0, -1.6], [-6.1, -0.9], [-6.0, -0.2], [-6.0, 0.5], [-6.1, 1.2], [-6.0, 1.9]]),
        target_fn=_line_target(np.array([12.0, 0.0]), np.zeros(2)),
        params=replace(base, horizon=60.0, k_r=0.0, k_t=1.8, r0=2.0, eta_min=0.08, sigma_omega=5.0, d_agent_safe=0.4, d_obs_safe=0.3),
        success_mode="corridor_pass",
        corridor_exit_x=5.2,
    )
    s4 = Scenario(
        key="s4_narrow_passage",
        title="S4 Narrow Passage",
        description="Backward-compatible alias of the medium corridor.",
        n_agents=6,
        obstacles=obstacles_s4_medium,
        initial_positions=np.array([[-6.0, -1.5], [-6.1, -0.9], [-6.0, -0.3], [-6.0, 0.3], [-6.1, 0.9], [-6.0, 1.5]]),
        target_fn=_line_target(np.array([12.0, 0.0]), np.zeros(2)),
        params=replace(base, horizon=40.0, k_r=0.0, k_t=1.2, r0=1.6, eta_min=0.06, sigma_omega=8.0, d_agent_safe=0.34, d_obs_safe=0.28, k_s=3.4),
        success_mode="corridor_pass",
        corridor_exit_x=5.2,
    )

    s5_stage0 = Scenario(
        key="s5_tracking_stage0",
        title="S5 Stage 0 No-Obstacle Tracking",
        description="Moving target without obstacles, used to isolate prediction and encirclement behavior.",
        n_agents=6,
        obstacles=[],
        initial_positions=np.array([[-7.0, -2.3], [-7.1, -1.4], [-7.0, -0.5], [-7.0, 0.4], [-7.1, 1.3], [-7.0, 2.2]]),
        target_fn=_turning_target(np.array([8.5, -0.2]), speed=0.15, omega=0.035),
        params=replace(base, horizon=65.0, t_pred=0.7, d_agent_safe=0.45, r_c=3.0, beta_lead=0.25, k_s=4.0, k_t=1.0, r0=1.8),
        success_mode="target_track",
    )
    s5_stage1 = Scenario(
        key="s5_tracking_stage1",
        title="S5 Stage 1 Single-Obstacle Tracking",
        description="Moving target with one obstacle, used to study first-order avoidance and tracking conflict.",
        n_agents=6,
        obstacles=[Obstacle(center=np.array([2.5, 0.0]), radius=0.9)],
        initial_positions=np.array([[-7.0, -2.3], [-7.1, -1.4], [-7.0, -0.5], [-7.0, 0.4], [-7.1, 1.3], [-7.0, 2.2]]),
        target_fn=_turning_target(np.array([8.5, -0.2]), speed=0.16, omega=0.035),
        params=replace(base, horizon=68.0, t_pred=0.7, d_obs_safe=0.3, d_agent_safe=0.45, r_c=3.0, beta_lead=0.25, k_s=4.0, k_t=1.0, r0=1.8),
        success_mode="target_track",
    )
    s5_stage2 = Scenario(
        key="s5_tracking_stage2",
        title="S5 Stage 2 Sparse-Obstacle Tracking",
        description="Moving target with two to three obstacles before entering medium-density tracking.",
        n_agents=6,
        obstacles=[
            Obstacle(center=np.array([0.8, 1.5]), radius=0.8),
            Obstacle(center=np.array([3.2, -1.4]), radius=0.85),
            Obstacle(center=np.array([5.8, 1.2]), radius=0.8),
        ],
        initial_positions=np.array([[-7.0, -2.3], [-7.1, -1.4], [-7.0, -0.5], [-7.0, 0.4], [-7.1, 1.3], [-7.0, 2.2]]),
        target_fn=_turning_target(np.array([8.5, -0.2]), speed=0.18, omega=0.04),
        params=replace(base, horizon=70.0, t_pred=0.7, d_obs_safe=0.32, d_agent_safe=0.46, r_c=3.0, beta_lead=0.28, k_s=4.0, k_t=1.0, r0=1.8),
        success_mode="target_track",
    )

    s5_easy = Scenario(
        key="s5_dense_tracking_easy",
        title="S5 Easy Obstacle Tracking",
        description="Low-density moving-target tracking used as a feasibility rung before dense tracking.",
        n_agents=6,
        obstacles=[
            Obstacle(center=np.array([0.5, 1.8]), radius=0.9),
            Obstacle(center=np.array([3.2, -1.7]), radius=0.9),
            Obstacle(center=np.array([6.0, 1.4]), radius=0.85),
        ],
        initial_positions=np.array([[-7.0, -1.8], [-7.2, -1.0], [-7.1, -0.2], [-7.0, 0.6], [-7.1, 1.4], [-7.3, 2.2]]),
        target_fn=_turning_target(np.array([8.5, -0.2]), speed=0.25, omega=0.06),
        params=replace(base, horizon=75.0, t_pred=0.7, d_obs_safe=0.35, d_agent_safe=0.55),
        success_mode="target_track",
    )

    s5_medium = Scenario(
        key="s5_dense_tracking_medium",
        title="S5 Medium Obstacle Tracking",
        description="Medium-density moving-target tracking for main dense-environment comparisons.",
        n_agents=6,
        obstacles=[
            Obstacle(center=np.array([-1.0, 2.2]), radius=0.95),
            Obstacle(center=np.array([0.2, -2.0]), radius=1.0),
            Obstacle(center=np.array([2.8, 1.8]), radius=1.0),
            Obstacle(center=np.array([4.4, -2.1]), radius=0.95),
            Obstacle(center=np.array([6.8, 0.6]), radius=1.05),
        ],
        initial_positions=np.array([[-7.0, -2.0], [-7.2, -1.1], [-7.1, -0.2], [-7.0, 0.7], [-7.1, 1.6], [-7.3, 2.5]]),
        target_fn=_turning_target(np.array([9.0, -0.2]), speed=0.35, omega=0.09),
        params=replace(base, horizon=75.0, t_pred=0.8, d_obs_safe=0.4, d_agent_safe=0.6),
        success_mode="target_track",
    )

    s5_hard = Scenario(
        key="s5_dense_tracking_hard",
        title="S5 Hard Dense-Obstacle Tracking",
        description="High-density moving-target pursuit retained as a robustness stress test.",
        n_agents=6,
        obstacles=[
            Obstacle(center=np.array([-1.5, 2.5]), radius=1.0),
            Obstacle(center=np.array([-0.5, -2.4]), radius=1.1),
            Obstacle(center=np.array([2.5, 2.0]), radius=1.1),
            Obstacle(center=np.array([2.3, -2.5]), radius=1.0),
            Obstacle(center=np.array([5.5, 0.2]), radius=1.4),
            Obstacle(center=np.array([7.8, 2.6]), radius=1.1),
            Obstacle(center=np.array([7.9, -2.7]), radius=1.0),
            Obstacle(center=np.array([10.0, 0.0]), radius=1.2),
        ],
        initial_positions=np.array([[-7.0, -2.2], [-7.2, -1.1], [-7.1, -0.1], [-7.0, 0.9], [-7.1, 2.0], [-7.3, 3.0]]),
        target_fn=_turning_target(np.array([9.5, -0.2]), speed=0.45, omega=0.12),
        params=replace(base, horizon=75.0, t_pred=0.9),
        success_mode="target_track",
    )

    s6 = Scenario(
        key="s6_fast_target",
        title="S6 Fast Target Interception",
        description="Open-space moving target used to validate predictive navigation.",
        n_agents=6,
        obstacles=[],
        initial_positions=np.array([[-8.0, -2.0], [-8.1, -1.0], [-8.0, 0.0], [-7.9, 1.0], [-8.0, 2.0], [-8.1, 3.0]]),
        target_fn=_line_target(np.array([0.0, 0.0]), np.array([0.75, 0.2])),
        params=replace(base, horizon=60.0, t_pred=1.0),
        success_mode="target_track",
    )
    return {
        scene.key: scene
        for scene in [
            s1,
            s2,
            s3,
            s4,
            s4_easy,
            s4_medium,
            s4_hard,
            s5_stage0,
            s5_stage1,
            s5_stage2,
            s5_easy,
            s5_medium,
            s5_hard,
            s6,
        ]
    }
