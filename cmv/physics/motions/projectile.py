from __future__ import annotations
import numpy as np
from cmv.physics.params import ProjectileParams


def _ground_hit(t, state):
    return state[1]

_ground_hit.terminal = True
_ground_hit.direction = -1


class ProjectileODE:
    name = "projectile"

    def __init__(self, params: ProjectileParams) -> None:
        self.p = params
        self.events = [_ground_hit]

    def equations(self, t: float, state: list[float]) -> list[float]:
        x, y, vx, vy = state
        p = self.p
        v = np.sqrt(vx**2 + vy**2)
        drag = p.k / p.m * v if v > 0 else 0.0
        return [vx, vy, -drag * vx, -p.g - drag * vy]

    def initial_state(self) -> list[float]:
        p = self.p
        rad = np.radians(p.angle_deg)
        return [p.x0, p.y0, p.v0 * np.cos(rad), p.v0 * np.sin(rad)]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.column_stack([sol_y[0], sol_y[1]])
        velocities = np.column_stack([sol_y[2], sol_y[3]])
        return positions, velocities

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        return np.array([deriv[2], deriv[3]])

    def energy(self, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy = state
        KE = 0.5 * self.p.m * (vx**2 + vy**2)
        PE = self.p.m * self.p.g * y
        return np.array([KE, PE, KE + PE])
