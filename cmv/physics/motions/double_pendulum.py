from __future__ import annotations
import numpy as np
from cmv.physics.params import DoublePendulumParams


class DoublePendulumODE:
    name = "double_pendulum"
    _method = "DOP853"

    def __init__(self, params: DoublePendulumParams) -> None:
        self.p = params

    def equations(self, t: float, state: list[float]) -> list[float]:
        th1, th2, w1, w2 = state
        p = self.p
        dth = th2 - th1
        m_sum = p.m1 + p.m2

        denom1 = p.L1 * (2 * p.m1 + p.m2 - p.m2 * np.cos(2 * dth))
        denom1 = max(abs(denom1), 1e-10) * np.sign(denom1) if denom1 != 0 else 1e-10

        denom2 = p.L2 * (2 * p.m1 + p.m2 - p.m2 * np.cos(2 * dth))
        denom2 = max(abs(denom2), 1e-10) * np.sign(denom2) if denom2 != 0 else 1e-10

        dw1 = (
            -p.g * m_sum * np.sin(th1)
            - p.m2 * p.g * np.sin(th1 - 2 * th2)
            - 2 * np.sin(dth) * p.m2 * (w2**2 * p.L2 + w1**2 * p.L1 * np.cos(dth))
        ) / denom1

        dw2 = (
            2 * np.sin(dth) * (
                m_sum * w1**2 * p.L1
                + p.g * m_sum * np.cos(th1)
                + w2**2 * p.L2 * p.m2 * np.cos(dth)
            )
        ) / denom2

        return [w1, w2, dw1, dw2]

    def initial_state(self) -> list[float]:
        p = self.p
        return [p.theta1_0, p.theta2_0, p.omega1_0, p.omega2_0]

    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        th1, th2 = sol_y[0], sol_y[1]
        w1, w2 = sol_y[2], sol_y[3]
        p = self.p

        x1 = p.L1 * np.sin(th1)
        y1 = -p.L1 * np.cos(th1)
        x2 = x1 + p.L2 * np.sin(th2)
        y2 = y1 - p.L2 * np.cos(th2)

        vx2 = p.L1 * w1 * np.cos(th1) + p.L2 * w2 * np.cos(th2)
        vy2 = p.L1 * w1 * np.sin(th1) + p.L2 * w2 * np.sin(th2)

        positions = np.column_stack([x2, y2])
        velocities = np.column_stack([vx2, vy2])
        return positions, velocities

    def deriv_to_accel(self, deriv: list[float], state: list[float]) -> np.ndarray:
        th1, th2 = state[0], state[1]
        w1, w2 = state[2], state[3]
        dw1, dw2 = deriv[2], deriv[3]
        p = self.p

        ax = (-p.L1 * (dw1 * np.sin(th1) + w1**2 * np.cos(th1))
              - p.L2 * (dw2 * np.sin(th2) + w2**2 * np.cos(th2)))
        ay = (p.L1 * (dw1 * np.cos(th1) - w1**2 * np.sin(th1))
              + p.L2 * (dw2 * np.cos(th2) - w2**2 * np.sin(th2)))
        return np.array([ax, ay])

    def energy(self, state: np.ndarray) -> np.ndarray:
        th1, th2, w1, w2 = state
        p = self.p
        x1 = p.L1 * np.sin(th1)
        y1 = -p.L1 * np.cos(th1)
        x2 = x1 + p.L2 * np.sin(th2)
        y2 = y1 - p.L2 * np.cos(th2)

        vx1 = p.L1 * w1 * np.cos(th1)
        vy1 = p.L1 * w1 * np.sin(th1)
        vx2 = vx1 + p.L2 * w2 * np.cos(th2)
        vy2 = vy1 + p.L2 * w2 * np.sin(th2)

        KE = 0.5 * p.m1 * (vx1**2 + vy1**2) + 0.5 * p.m2 * (vx2**2 + vy2**2)
        PE = p.m1 * p.g * y1 + p.m2 * p.g * y2
        return np.array([KE, PE, KE + PE])
