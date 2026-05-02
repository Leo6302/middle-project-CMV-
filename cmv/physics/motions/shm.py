from __future__ import annotations
import numpy as np
from cmv.physics.params import SHMParams


class SHMODE:
    name = "shm"

    def __init__(self, params: SHMParams) -> None:
        self.p = params

    def equations(self, t: float, state: list[float]) -> list[float]:
        x, v = state
        p = self.p
        omega0_sq = p.k / p.m
        dv = -omega0_sq * x - (p.gamma / p.m) * v + (p.F0 / p.m) * np.cos(p.omega_d * t)
        return [v, dv]

    def initial_state(self) -> list[float]:
        return [self.p.x0, self.p.v0]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = sol_y[0]
        v = sol_y[1]
        y = np.zeros_like(x)
        vy = np.zeros_like(v)
        positions = np.column_stack([x, y])
        velocities = np.column_stack([v, vy])
        return positions, velocities

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        return np.array([deriv[1], 0.0])

    def energy(self, state: np.ndarray) -> np.ndarray:
        x, v = state[0], state[1]
        KE = 0.5 * self.p.m * v**2
        PE = 0.5 * self.p.k * x**2
        return np.array([KE, PE, KE + PE])
