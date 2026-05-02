from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np


@dataclass
class PendulumParams:
    L: float = 1.0
    m: float = 1.0
    g: float = 9.81
    theta0: float = 0.5
    omega0: float = 0.0
    gamma: float = 0.0
    t_span: tuple = (0.0, 10.0)
    n_points: int = 1000

    SLIDER_RANGES: ClassVar[dict] = {
        "L":      (0.1, 5.0),
        "m":      (0.1, 10.0),
        "g":      (0.1, 20.0),
        "theta0": (-np.pi, np.pi),
        "gamma":  (0.0, 2.0),
    }


@dataclass
class DoublePendulumParams:
    L1: float = 1.0
    L2: float = 1.0
    m1: float = 1.0
    m2: float = 1.0
    g: float = 9.81
    theta1_0: float = 1.57
    theta2_0: float = 0.0
    omega1_0: float = 0.0
    omega2_0: float = 0.0
    t_span: tuple = (0.0, 15.0)
    n_points: int = 2000

    SLIDER_RANGES: ClassVar[dict] = {
        "theta1_0": (-np.pi, np.pi),
        "theta2_0": (-np.pi, np.pi),
        "L1":       (0.2, 3.0),
        "L2":       (0.2, 3.0),
        "g":        (0.1, 20.0),
    }


@dataclass
class ProjectileParams:
    v0: float = 20.0
    angle_deg: float = 45.0
    m: float = 1.0
    g: float = 9.81
    k: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    t_span: tuple = (0.0, 10.0)
    n_points: int = 1000

    SLIDER_RANGES: ClassVar[dict] = {
        "v0":        (1.0, 50.0),
        "angle_deg": (1.0, 89.0),
        "k":         (0.0, 1.0),
        "g":         (0.1, 20.0),
    }


@dataclass
class SHMParams:
    k: float = 10.0
    m: float = 1.0
    x0: float = 1.0
    v0: float = 0.0
    gamma: float = 0.0
    F0: float = 0.0
    omega_d: float = 3.0
    t_span: tuple = (0.0, 20.0)
    n_points: int = 2000

    SLIDER_RANGES: ClassVar[dict] = {
        "k":       (0.1, 50.0),
        "m":       (0.1, 10.0),
        "gamma":   (0.0, 5.0),
        "F0":      (0.0, 5.0),
        "omega_d": (0.1, 10.0),
    }


@dataclass
class CircularParams:
    r: float = 1.0
    omega: float = 1.0
    t_span: tuple = (0.0, 10.0)
    n_points: int = 500

    SLIDER_RANGES: ClassVar[dict] = {
        "r":     (0.1, 5.0),
        "omega": (0.1, 5.0),
    }


@dataclass
class KeplerParams:
    GM: float = 39.478
    r0: float = 1.0
    vt0: float = 6.283
    ecc: float = 0.0
    t_span: tuple = (0.0, 5.0)
    n_points: int = 2000

    SLIDER_RANGES: ClassVar[dict] = {
        "ecc":  (0.0, 0.95),
        "GM":   (1.0, 100.0),
        "r0":   (0.5, 5.0),
    }


# ── 3D motion params ──────────────────────────────────────────────────────

@dataclass
class SphericalPendulumParams:
    L: float = 1.0
    m: float = 1.0
    g: float = 9.81
    theta0: float = 0.8    # polar angle from vertical [rad]
    phi0: float = 0.0      # azimuthal angle [rad]
    dtheta0: float = 0.0
    dphi0: float = 1.5     # initial azimuthal spin
    t_span: tuple = (0.0, 20.0)
    n_points: int = 2000

    SLIDER_RANGES: ClassVar[dict] = {
        "L":       (0.2, 3.0),
        "g":       (0.1, 20.0),
        "theta0":  (0.05, np.pi - 0.05),
        "dphi0":   (-5.0, 5.0),
        "dtheta0": (-3.0, 3.0),
    }


@dataclass
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 2.667
    x0: float = 1.0
    y0: float = 0.0
    z0: float = 0.0
    t_span: tuple = (0.0, 40.0)
    n_points: int = 5000

    SLIDER_RANGES: ClassVar[dict] = {
        "sigma": (1.0, 20.0),
        "rho":   (1.0, 50.0),
        "beta":  (0.1, 6.0),
    }


@dataclass
class MagneticParticleParams:
    m: float = 1.0
    q: float = 1.0
    Bz: float = 1.0    # magnetic field strength (z-direction)
    Ez: float = 0.0    # electric field (z-direction)
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    vx0: float = 1.0
    vy0: float = 0.0
    vz0: float = 0.5   # helical drift
    t_span: tuple = (0.0, 20.0)
    n_points: int = 2000

    SLIDER_RANGES: ClassVar[dict] = {
        "Bz":  (0.1, 5.0),
        "vz0": (-3.0, 3.0),
        "vx0": (0.1, 5.0),
        "Ez":  (-2.0, 2.0),
    }
