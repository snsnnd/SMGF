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
    env_flow: Callable[[np.ndarray, float], np.ndarray] = field(
        default=lambda p, t: np.zeros(2, dtype=float)
    )


@dataclass(frozen=True)
class Method:
    name: str
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
    }


class SMGFController:
    def __init__(self, params: Params, method: Method, n_agents: int):
        self.params = params
        self.method = method
        self.prev_sep = np.tile(np.array([1.0, 0.0]), (n_agents, n_agents, 1))
        self.prev_curl_sign = np.array(
            [params.default_curl_sign if i % 2 == 0 else -params.default_curl_sign for i in range(n_agents)],
            dtype=float,
        )

    def _predicted_target(self, target: Target) -> np.ndarray:
        horizon = min(self.params.t_pred, self.params.t_pred_max) if self.method.allow_prediction else 0.0
        return target.position + horizon * target.velocity

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

    def compute(self, positions: np.ndarray, target: Target, omega_prev: np.ndarray, obstacles: list[Obstacle], t: float) -> dict[str, np.ndarray]:
        self.params_obstacles = obstacles
        n_agents = len(positions)
        predicted_target = self._predicted_target(target)
        neighbors = self._neighbor_sets(positions)
        u = np.zeros_like(positions)
        omega = np.zeros(n_agents)
        rho = np.zeros(n_agents)
        psi_hat = np.zeros(n_agents)
        psi_tilde = np.zeros(n_agents)
        phi_vals = np.zeros(n_agents)
        for i, position in enumerate(positions):
            if self.method.traditional_apf:
                nav = -self.params.w_g * (position - target.position)
                rep = sum((self._traditional_repulsion(position, obs) for obs in obstacles), start=np.zeros(2))
                raw = (nav + rep) / self.params.gamma_d
                u[i] = smooth_bound(raw, self.params.u_max)
                omega[i] = omega_prev[i]
                rho[i] = 0.0
                psi_hat[i] = 1.0
                psi_tilde[i] = 1.0
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
            psi_hat[i], psi_tilde[i], d_val = self._local_psi(i, positions, target, neighbors[i], n_agents)
            alpha = min(1.0, self.params.dt / max(self.params.tau_omega, self.params.dt))
            omega[i] = clip01((1.0 - alpha) * omega_prev[i] + alpha * phi)
            phi_vals[i] = phi
            eta = 1.0 if not self.method.use_omega else self.params.eta_min + (1.0 - self.params.eta_min) * np.exp(-self.params.sigma_omega * omega[i])
            if self.method.force_rho_one:
                rho[i] = 1.0
            else:
                rho[i] = eta * d_val
            total_topo = topo if self.method.force_rho_one else rho[i] * topo
            raw = (nav + rep + curl + safe + total_topo + enc + ang) / self.params.gamma_d
            u[i] = smooth_bound(raw, self.params.u_max)

        env = np.array([self.params.env_flow(p, t) for p in positions])
        return {
            "u": u,
            "omega": omega,
            "rho": rho,
            "psi_hat": psi_hat,
            "psi_tilde": psi_tilde,
            "phi": phi_vals,
            "env": env,
            "predicted_target": predicted_target,
        }


def with_horizon(params: Params, horizon: float) -> Params:
    return replace(params, horizon=horizon)
