from __future__ import annotations
import numpy as np
from cmv.physics.params import CircularParams


class CircularODE:
    name = "circular"

    def __init__(self, params: CircularParams) -> None:
        self.p = params

    def equations(self, t: float, state: list[float]) -> list[float]:
        x, y, vx, vy = state
        p = self.p
        r = np.sqrt(x**2 + y**2)
        r = max(r, 1e-12)
        # Centripetal: a = -omega^2 * r_vec
        ax = -p.omega**2 * x
        ay = -p.omega**2 * y
        return [vx, vy, ax, ay]

    def initial_state(self) -> list[float]:
        p = self.p
        return [p.r, 0.0, 0.0, p.r * p.omega]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.column_stack([sol_y[0], sol_y[1]])
        velocities = np.column_stack([sol_y[2], sol_y[3]])
        return positions, velocities

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        return np.array([deriv[2], deriv[3]])

    def energy(self, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy = state
        # No potential energy for uniform circular motion (external force)
        KE = 0.5 * (vx**2 + vy**2)
        PE = 0.0
        return np.array([KE, PE, KE + PE])
