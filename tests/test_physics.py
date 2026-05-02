import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from cmv.physics import (
    PendulumParams, DoublePendulumParams, ProjectileParams,
    SHMParams, KeplerParams, CircularParams,
    PendulumODE, DoublePendulumODE, ProjectileODE,
    SHMODE, KeplerODE, CircularODE,
    SimulationEngine,
)


def make_engine(ode_cls, param_cls, **kwargs):
    params = param_cls(**kwargs)
    ode = ode_cls(params)
    return SimulationEngine(ode, params)


def test_energy_conservation_pendulum():
    eng = make_engine(PendulumODE, PendulumParams, gamma=0.0, t_span=(0, 20), n_points=2000)
    result = eng.run()
    E = result.energies[:, 2]
    drift = abs((E[-1] - E[0]) / (abs(E[0]) + 1e-12))
    assert drift < 1e-2, f"Energy drift: {drift:.2e}"


def test_pendulum_positions_finite():
    eng = make_engine(PendulumODE, PendulumParams)
    result = eng.run()
    assert np.all(np.isfinite(result.positions))
    assert np.all(np.isfinite(result.velocities))


def test_double_pendulum_no_diverge():
    eng = make_engine(DoublePendulumODE, DoublePendulumParams,
                      t_span=(0, 10), n_points=1000)
    result = eng.run()
    assert np.all(np.isfinite(result.positions))
    assert np.all(np.abs(result.positions) < 1e6)


def test_kepler_angular_momentum():
    eng = make_engine(KeplerODE, KeplerParams, ecc=0.0, t_span=(0, 5), n_points=2000)
    result = eng.run()
    x, y = result.positions[:, 0], result.positions[:, 1]
    vx, vy = result.velocities[:, 0], result.velocities[:, 1]
    L = x * vy - y * vx
    assert np.allclose(L, L[0], rtol=5e-3), "Angular momentum not conserved"


def test_projectile_range_no_drag():
    eng = make_engine(ProjectileODE, ProjectileParams,
                      v0=10.0, angle_deg=45.0, k=0.0, t_span=(0, 5), n_points=2000)
    result = eng.run()
    max_x = result.positions[:, 0].max()
    expected = 10.0**2 / 9.81
    assert abs(max_x - expected) < 0.5, f"max_x={max_x:.3f}, expected={expected:.3f}"


def test_shm_period():
    k, m = 10.0, 1.0
    omega0 = np.sqrt(k / m)
    T = 2 * np.pi / omega0
    eng = make_engine(SHMODE, SHMParams, k=k, m=m, gamma=0.0,
                      x0=1.0, v0=0.0, F0=0.0, t_span=(0, 3 * T), n_points=1000)
    result = eng.run()
    # At t ≈ T, x should return close to x0
    t = result.timestamps
    idx = np.argmin(np.abs(t - T))
    assert abs(result.positions[idx, 0] - 1.0) < 0.05, "SHM period mismatch"


def test_circular_radius_constant():
    eng = make_engine(CircularODE, CircularParams, r=1.0, omega=1.0,
                      t_span=(0, 10), n_points=500)
    result = eng.run()
    r = np.sqrt(result.positions[:, 0]**2 + result.positions[:, 1]**2)
    assert np.allclose(r, 1.0, atol=0.05), "Circular radius not constant"
