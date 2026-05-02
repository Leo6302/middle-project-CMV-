import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import tempfile
from pathlib import Path

from cmv.physics import PendulumParams, PendulumODE, SimulationEngine
from cmv.data.recorder import DataRecorder, DataBundle, _SimpleScaler
from cmv.ml.model import PhysicsMLPModel
from cmv.ml.trainer import MLTrainer


def make_test_bundle(n=500):
    params = PendulumParams(gamma=0.3, t_span=(0, 10), n_points=n)
    ode = PendulumODE(params)
    eng = SimulationEngine(ode, params)
    result = eng.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = DataRecorder(Path(tmpdir))
        save_path = recorder.save(result)
        bundle = recorder.load_as_bundle(save_path)

    return bundle


def test_scaler_normalize():
    s = _SimpleScaler()
    X = np.random.randn(100, 5) * 3 + 2
    Xn = s.fit_transform(X)
    assert abs(Xn.mean()) < 0.1
    assert abs(Xn.std() - 1.0) < 0.1


def test_data_bundle_shapes():
    bundle = make_test_bundle(500)
    n_total = 500
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    assert bundle.X_train.shape[0] == n_train
    assert bundle.X_val.shape[0] == n_val
    assert bundle.X_train.shape[1] == 4  # [x, y, vx, vy]
    assert bundle.y_train.shape[1] == 2  # [ax, ay]


def test_model_forward():
    model = PhysicsMLPModel(input_dim=4, output_dim=2)
    import torch
    X = torch.randn(16, 4)
    out = model(X)
    assert out.shape == (16, 2)


def test_training_converges():
    bundle = make_test_bundle(500)
    model = PhysicsMLPModel(input_dim=4, output_dim=2)
    trainer = MLTrainer(model, bundle, epochs=30, lr=1e-3)
    history = trainer.train()
    assert history["train"][-1] < history["train"][0], "Train loss should decrease"
    assert len(history["train"]) == 30


def test_predict_physical_units():
    bundle = make_test_bundle(500)
    model = PhysicsMLPModel(input_dim=4, output_dim=2)
    trainer = MLTrainer(model, bundle, epochs=20, lr=1e-3)
    trainer.train()

    X_raw = bundle.scaler_X.inverse_transform(bundle.X_test[:10])
    X_scaled = bundle.scaler_X.transform(X_raw)
    a_pred = trainer.predict(X_scaled)
    assert a_pred.shape == (10, 2)
    assert np.all(np.isfinite(a_pred))


def test_save_load(tmp_path):
    bundle = make_test_bundle(300)
    model = PhysicsMLPModel(input_dim=4, output_dim=2)
    trainer = MLTrainer(model, bundle, epochs=5)
    trainer.train()
    path = tmp_path / "test.pt"
    trainer.save(path)
    assert path.exists()
    trainer.load(path)
