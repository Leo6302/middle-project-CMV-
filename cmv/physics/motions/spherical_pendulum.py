from __future__ import annotations
import numpy as np
from cmv.physics.params import SphericalPendulumParams


class SphericalPendulumODE:
    name = "spherical_pendulum"
    is_3d = True

    def __init__(self, params: SphericalPendulumParams) -> None:
        self.p = params

    def equations(self, t: float, state: list[float]) -> list[float]:
        theta, phi, dtheta, dphi = state
        p = self.p
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)

        ddtheta = sin_th * cos_th * dphi**2 - (p.g / p.L) * sin_th
        if abs(sin_th) < 1e-8:
            ddphi = 0.0
        else:
            ddphi = -2.0 * cos_th / sin_th * dtheta * dphi

        return [dtheta, dphi, ddtheta, ddphi]

    def initial_state(self) -> list[float]:
        p = self.p
        return [p.theta0, p.phi0, p.dtheta0, p.dphi0]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta, phi = sol_y[0], sol_y[1]
        dtheta, dphi = sol_y[2], sol_y[3]
        p = self.p
        sin_th, cos_th = np.sin(theta), np.cos(theta)
        sin_ph, cos_ph = np.sin(phi), np.cos(phi)

        x = p.L * sin_th * cos_ph
        y = p.L * sin_th * sin_ph
        z = -p.L * cos_th

        vx = p.L * (dtheta * cos_th * cos_ph - sin_th * sin_ph * dphi)
        vy = p.L * (dtheta * cos_th * sin_ph + sin_th * cos_ph * dphi)
        vz = p.L * dtheta * sin_th

        return np.column_stack([x, y, z]), np.column_stack([vx, vy, vz])

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        theta, phi, dtheta, dphi = state
        _, _, ddtheta, ddphi = deriv
        p = self.p
        sin_th, cos_th = np.sin(theta), np.cos(theta)
        sin_ph, cos_ph = np.sin(phi), np.cos(phi)

        # d²r/dt² via chain rule on r = L*(sinθ cosφ, sinθ sinφ, -cosθ)
        ax = p.L * (ddtheta * cos_th * cos_ph
                    - dtheta**2 * sin_th * cos_ph
                    - 2 * dtheta * dphi * cos_th * sin_ph
                    - ddphi * sin_th * sin_ph
                    - dphi**2 * sin_th * cos_ph)
        ay = p.L * (ddtheta * cos_th * sin_ph
                    - dtheta**2 * sin_th * sin_ph
                    + 2 * dtheta * dphi * cos_th * cos_ph
                    + ddphi * sin_th * cos_ph
                    - dphi**2 * sin_th * sin_ph)
        az = p.L * (ddtheta * sin_th + dtheta**2 * cos_th)
        return np.array([ax, ay, az])

    def energy(self, state: np.ndarray) -> np.ndarray:
        theta, phi, dtheta, dphi = state
        p = self.p
        KE = 0.5 * p.m * p.L**2 * (dtheta**2 + np.sin(theta)**2 * dphi**2)
        PE = p.m * p.g * p.L * (1 - np.cos(theta))
        return np.array([KE, PE, KE + PE])
