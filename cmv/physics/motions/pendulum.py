from __future__ import annotations
import numpy as np
from cmv.physics.params import PendulumParams


class PendulumODE:
    name = "pendulum"

    def __init__(self, params: PendulumParams) -> None:
        self.p = params

    def equations(self, t: float, state: list[float]) -> list[float]:
        theta, omega = state
        dtheta = omega
        domega = -(self.p.g / self.p.L) * np.sin(theta) - self.p.gamma * omega
        return [dtheta, domega]

    def initial_state(self) -> list[float]:
        return [self.p.theta0, self.p.omega0]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = sol_y[0]
        omega = sol_y[1]
        x = self.p.L * np.sin(theta)
        y = -self.p.L * np.cos(theta)
        vx = self.p.L * omega * np.cos(theta)
        vy = self.p.L * omega * np.sin(theta)
        positions = np.column_stack([x, y])
        velocities = np.column_stack([vx, vy])
        return positions, velocities

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        theta, omega = state
        domega = deriv[1]
        ax = self.p.L * (domega * np.cos(theta) - omega**2 * np.sin(theta))
        ay = self.p.L * (domega * np.sin(theta) + omega**2 * np.cos(theta))
        return np.array([ax, ay])

    def energy(self, state: np.ndarray) -> np.ndarray:
        theta, omega = state[0], state[1]
        KE = 0.5 * self.p.m * self.p.L**2 * omega**2
        PE = self.p.m * self.p.g * self.p.L * (1 - np.cos(theta))
        return np.array([KE, PE, KE + PE])
