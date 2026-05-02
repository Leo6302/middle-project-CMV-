from __future__ import annotations
from enum import Enum
import numpy as np


class CoordType(Enum):
    CARTESIAN = "cartesian"


class CoordinateSystem:
    def __init__(self, coord_type: CoordType = CoordType.CARTESIAN) -> None:
        self.coord_type = coord_type

    def transform(self, positions: np.ndarray) -> np.ndarray:
        return positions

    def inverse_transform(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def axis_labels(self) -> tuple[str, str, str]:
        return ("x [m]", "y [m]", "z [m]")

    def axis_limits(self, coords: np.ndarray, margin: float = 0.1) -> list[tuple[float, float]]:
        limits = []
        for i in range(min(3, coords.shape[1])):
            lo, hi = coords[:, i].min(), coords[:, i].max()
            span = max(hi - lo, 1e-3)
            limits.append((lo - margin * span, hi + margin * span))
        return limits
