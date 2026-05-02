from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from cmv.physics.base import SimResult, MotionODE


class SimulationEngine:
    def __init__(self, motion: MotionODE, params) -> None:
        self.motion = motion
        self.params = params

    def run(
        self,
        t_span: tuple[float, float] | None = None,
        n_points: int | None = None,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        y0_override: np.ndarray | None = None,
    ) -> SimResult:
        p = self.params
        t0, tf = t_span or p.t_span
        n = n_points or p.n_points
        m = method or getattr(p, "_method", "RK45")

        t_eval = np.linspace(t0, tf, n)
        y0 = list(y0_override) if y0_override is not None else self.motion.initial_state()

        events = getattr(self.motion, "events", None)
        sol = solve_ivp(
            self.motion.equations,
            (t0, tf),
            y0,
            method=m,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            events=events,
            dense_output=False,
        )

        t_out = sol.t
        positions, velocities = self.motion.to_cartesian(sol.y)

        # vectorised: numerical derivative of velocity → acceleration
        accelerations = np.gradient(velocities, t_out, axis=0)

        # vectorised: iterate columns (cache-friendly) instead of a Python loop
        energies = np.array([self.motion.energy(col) for col in sol.y.T])

        dt = float(np.mean(np.diff(t_out))) if len(t_out) > 1 else (tf - t0) / n

        return SimResult(
            motion_type=self.motion.name,
            timestamps=t_out,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            energies=energies,
            params=vars(p).copy(),
            dt=dt,
            raw_y=sol.y[:, -1],
        )

    def update_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self.params, k, v)
