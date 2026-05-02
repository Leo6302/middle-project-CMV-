import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import tempfile
from pathlib import Path

from cmv.physics import PendulumParams, PendulumODE, SimulationEngine
from cmv.data.recorder import DataRecorder
from cmv.ml.model import PhysicsMLPModel
from cmv.ml.trainer import MLTrainer
from cmv.viz.comparator import ResultComparator


def test_full_pipeline_pendulum(tmp_path):
    # 1. Simulation
    params = PendulumParams(gamma=0.3, t_span=(0, 10), n_points=500)
    ode = PendulumODE(params)
    result = SimulationEngine(ode, params).run()
    assert len(result.timestamps) > 0
    assert np.all(np.isfinite(result.positions))

    # 2. Save
    recorder = DataRecorder(tmp_path)
    save_path = recorder.save(result)
    assert (save_path / "trajectory.npy").exists()
    assert (save_path / "metadata.json").exists()

    # 3. Load as ML bundle
    bundle = recorder.load_as_bundle(save_path)
    assert bundle.X_train.shape[1] == 4
    assert bundle.y_train.shape[1] == 2

    # 4. Train
    model = PhysicsMLPModel(input_dim=4, output_dim=2)
    trainer = MLTrainer(model, bundle, epochs=20, lr=1e-3)
    history = trainer.train()
    assert len(history["train"]) == 20

    # 5. Compare
    comp = ResultComparator(trainer, result)
    t_end = result.timestamps[min(100, len(result.timestamps) - 1)]
    pred = comp.predict_trajectory(result.positions[0], (0.0, float(t_end)), n_points=50)
    assert np.all(np.isfinite(pred.positions))

    metrics = comp.compute_metrics(pred)
    amp = result.positions[:, 0].std()
    assert metrics.position_rmse < amp * 5.0  # Generous bound for 20-epoch training


def test_list_sessions(tmp_path):
    params = PendulumParams(t_span=(0, 5), n_points=200)
    ode = PendulumODE(params)
    result = SimulationEngine(ode, params).run()
    recorder = DataRecorder(tmp_path)
    recorder.save(result)
    sessions = recorder.list_sessions("pendulum")
    assert len(sessions) >= 1
