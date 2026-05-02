from __future__ import annotations
import numpy as np
from cmv.physics.params import LorenzParams


class LorenzODE:
    name = "lorenz"
    is_3d = True

    def __init__(self, params: LorenzParams) -> None:
        self.p = params

    def equations(self, t: float, state: list[float]) -> list[float]:
        x, y, z = state
        p = self.p
        dx = p.sigma * (y - x)
        dy = x * (p.rho - z) - y
        dz = x * y - p.beta * z
        return [dx, dy, dz]

    def initial_state(self) -> list[float]:
        p = self.p
        return [p.x0, p.y0, p.z0]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = sol_y[0], sol_y[1], sol_y[2]
        positions = np.column_stack([x, y, z])
        # Velocities = the Lorenz vector field evaluated at each point
        p = self.p
        vx = p.sigma * (y - x)
        vy = x * (p.rho - z) - y
        vz = x * y - p.beta * z
        velocities = np.column_stack([vx, vy, vz])
        return positions, velocities

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        # For 1st-order system, store the vector field as "acceleration"
        return np.array(deriv)

    def energy(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state[0], state[1], state[2]
        # Lorenz has no physical energy; use squared norm as proxy
        norm_sq = x**2 + y**2 + z**2
        KE = 0.5 * norm_sq
        PE = 0.0
        return np.array([KE, PE, KE])
