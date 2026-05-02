from __future__ import annotations
import numpy as np
from cmv.physics.params import KeplerParams


class KeplerODE:
    name = "kepler"

    def __init__(self, params: KeplerParams) -> None:
        self.p = params

    def equations(self, t: float, state: list[float]) -> list[float]:
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        r = max(r, 0.01)
        factor = -self.p.GM / r**3
        return [vx, vy, factor * x, factor * y]

    def initial_state(self) -> list[float]:
        p = self.p
        r0 = p.r0
        # Circular velocity at r0: v_c = sqrt(GM/r0)
        v_c = np.sqrt(p.GM / r0)
        # Adjust for eccentricity: v_t = v_c * sqrt(1 + ecc)
        vt = v_c * np.sqrt(1.0 + p.ecc)
        return [r0, 0.0, 0.0, vt]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.column_stack([sol_y[0], sol_y[1]])
        velocities = np.column_stack([sol_y[2], sol_y[3]])
        return positions, velocities

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        return np.array([deriv[2], deriv[3]])

    def energy(self, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        r = max(r, 1e-12)
        KE = 0.5 * (vx**2 + vy**2)
        PE = -self.p.GM / r
        return np.array([KE, PE, KE + PE])
