from __future__ import annotations
import numpy as np
from cmv.physics.params import MagneticParticleParams


class MagneticParticleODE:
    name = "magnetic_particle"
    is_3d = True

    def __init__(self, params: MagneticParticleParams) -> None:
        self.p = params

    def equations(self, t: float, state: list[float]) -> list[float]:
        x, y, z, vx, vy, vz = state
        p = self.p
        # Lorentz force: F = q(v × B), B = Bz * ẑ
        qm = p.q / p.m
        ax = qm * (vy * p.Bz)
        ay = qm * (-vx * p.Bz)
        az = qm * p.Ez   # optional electric field in z
        return [vx, vy, vz, ax, ay, az]

    def initial_state(self) -> list[float]:
        p = self.p
        return [p.x0, p.y0, p.z0, p.vx0, p.vy0, p.vz0]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.column_stack([sol_y[0], sol_y[1], sol_y[2]])
        velocities = np.column_stack([sol_y[3], sol_y[4], sol_y[5]])
        return positions, velocities

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        return np.array([deriv[3], deriv[4], deriv[5]])

    def energy(self, state: np.ndarray) -> np.ndarray:
        vx, vy, vz = state[3], state[4], state[5]
        KE = 0.5 * self.p.m * (vx**2 + vy**2 + vz**2)
        PE = 0.0
        return np.array([KE, PE, KE])
