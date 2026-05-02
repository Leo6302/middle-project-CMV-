from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np


@dataclass
class SimResult:
    motion_type: str
    timestamps: np.ndarray        # shape (N,)
    positions: np.ndarray         # shape (N, 2) or (N, 3)
    velocities: np.ndarray        # shape (N, 2) or (N, 3)
    accelerations: np.ndarray     # shape (N, 2) or (N, 3)
    energies: np.ndarray          # shape (N, 3) — [KE, PE, E_total]
    params: dict
    dt: float
    coord_system: str = "cartesian"
    metadata: dict = field(default_factory=dict)
    raw_y: np.ndarray = field(default=None)  # 마지막 ODE 상태벡터 (시뮬레이션 연장용)


@runtime_checkable
class MotionODE(Protocol):
    def equations(self, t: float, state: list[float]) -> list[float]: ...
    def energy(self, state: np.ndarray) -> np.ndarray: ...
    def initial_state(self) -> list[float]: ...
    def to_cartesian(self, sol_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
