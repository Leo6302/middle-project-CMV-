import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from cmv.coords.transforms import pad_to_3d
from cmv.coords.systems import CoordinateSystem, CoordType


rng = np.random.default_rng(42)


def test_pad_to_3d():
    pos2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    pos3 = pad_to_3d(pos2)
    assert pos3.shape == (2, 3)
    assert np.all(pos3[:, 2] == 0.0)


def test_coord_system_cartesian_identity():
    cs = CoordinateSystem(CoordType.CARTESIAN)
    pos = rng.uniform(-5, 5, (50, 3))
    assert np.allclose(cs.transform(pos), pos)


def test_coord_system_inverse_identity():
    cs = CoordinateSystem()
    pos = rng.uniform(-5, 5, (50, 3))
    assert np.allclose(cs.inverse_transform(pos), pos)


def test_coord_system_labels():
    cs = CoordinateSystem(CoordType.CARTESIAN)
    labels = cs.axis_labels()
    assert len(labels) == 3
    assert "x" in labels[0]
