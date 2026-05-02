from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from cmv.data.recorder import DataBundle
from cmv.ml.model import PhysicsMLPModel


class MLTrainer:
    def __init__(
        self,
        model: PhysicsMLPModel,
        bundle: DataBundle,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 512,
        weight_decay: float = 1e-5,
        device: str = "auto",
        rollout_weight: float = 0.5,
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model = model.to(device)
        self.bundle = bundle
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.rollout_weight = rollout_weight

        # Pre-build scaler tensors for rollout loss (avoids per-batch allocation)
        if rollout_weight > 0 and bundle.X_train_next is not None:
            sx = torch.tensor(bundle.scaler_X.scale_, dtype=torch.float32)
            mx = torch.tensor(bundle.scaler_X.mean_,  dtype=torch.float32)
            sy = torch.tensor(bundle.scaler_y.scale_, dtype=torch.float32)
            my = torch.tensor(bundle.scaler_y.mean_,  dtype=torch.float32)
            self._sx = sx.to(device); self._mx = mx.to(device)
            self._sy = sy.to(device); self._my = my.to(device)
            self._dt = float(bundle.metadata.get("dt", 0.01))
        else:
            self._sx = self._mx = self._sy = self._my = self._dt = None

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # patience=15: val_loss 15 에포크 미개선 시 LR 절반 (빠른 수렴)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )
        self.criterion = nn.MSELoss()

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_next: np.ndarray | None = None,
        shuffle: bool = True,
    ) -> DataLoader:
        tensors = [torch.tensor(X, dtype=torch.float32),
                   torch.tensor(y, dtype=torch.float32)]
        if X_next is not None:
            tensors.append(torch.tensor(X_next, dtype=torch.float32))
        pin = self.device == "cuda"
        return DataLoader(TensorDataset(*tensors), batch_size=self.batch_size,
                          shuffle=shuffle, pin_memory=pin, num_workers=0)

    def _rollout_loss(self, Xb: torch.Tensor, a_pred: torch.Tensor,
                      Xb_next: torch.Tensor) -> torch.Tensor:
        """1-step Euler rollout loss — works for any-dimensional state."""
        state  = Xb     * self._sx + self._mx
        a_phys = a_pred * self._sy + self._my
        dt     = self._dt

        n_feat = state.shape[1]
        n_tgt  = a_phys.shape[1]

        if n_feat == n_tgt:
            # 1st-order system (e.g. Lorenz): next = state + deriv*dt
            state_next_pred = state + a_phys * dt
        else:
            # 2nd-order system: state = [pos | vel], target = accel
            half  = n_feat // 2
            pos   = state[:, :half]
            vel   = state[:, half:]
            state_next_pred = torch.cat([pos + vel * dt,
                                         vel + a_phys * dt], dim=1)

        state_next_norm = (state_next_pred - self._mx) / self._sx
        return self.criterion(state_next_norm, Xb_next)

    def train(
        self,
        on_epoch_end: Optional[Callable[[int, float, float], None]] = None,
        val_target: Optional[float] = None,
        max_epochs: Optional[int] = None,
        stop_fn: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """
        val_target  : 수렴 판정 val loss 임계값 (None이면 체크 안 함)
        max_epochs  : 최대 에포크 상한 (None이면 self.epochs 사용)
        반환 dict에 'converged' (bool) 키 포함
        """
        b = self.bundle
        use_rollout = (self.rollout_weight > 0 and b.X_train_next is not None)
        train_loader = self._make_loader(
            b.X_train, b.y_train,
            X_next=b.X_train_next if use_rollout else None,
            shuffle=True,
        )
        val_loader = self._make_loader(
            b.X_val, b.y_val,
            X_next=b.X_val_next if use_rollout else None,
            shuffle=False,
        )

        total = max_epochs if max_epochs is not None else self.epochs
        history: dict = {"train": [], "val": [], "converged": False, "stopped": False}

        for epoch in range(1, total + 1):
            # ── train ──────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                Xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                a_pred = self.model(Xb)
                loss = self.criterion(a_pred, yb)
                if use_rollout:
                    Xb_next = batch[2].to(self.device)
                    loss = loss + self.rollout_weight * self._rollout_loss(Xb, a_pred, Xb_next)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(b.X_train)

            # ── validate ───────────────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    Xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                    a_pred = self.model(Xb)
                    vl = self.criterion(a_pred, yb)
                    if use_rollout:
                        Xb_next = batch[2].to(self.device)
                        vl = vl + self.rollout_weight * self._rollout_loss(Xb, a_pred, Xb_next)
                    val_loss += vl.item() * len(Xb)
            val_loss /= max(len(b.X_val), 1)

            self.scheduler.step(val_loss)   # ReduceLROnPlateau needs val_loss
            history["train"].append(train_loss)
            history["val"].append(val_loss)

            if on_epoch_end:
                on_epoch_end(epoch, train_loss, val_loss)

            # ── convergence check ──────────────────────────────────────────
            if val_target is not None and val_loss <= val_target:
                history["converged"] = True
                break

            # ── cancellation check ─────────────────────────────────────────
            if stop_fn is not None and stop_fn():
                history["stopped"] = True
                break

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions in physical units (inverse-normalized)."""
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(Xt).cpu().numpy()
        return self.bundle.scaler_y.inverse_transform(pred)

    def save(self, path: Path, extra_meta: dict | None = None) -> None:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        np.savez(
            path.parent / "scalers.npz",
            X_mean=self.bundle.scaler_X.mean_,
            X_scale=self.bundle.scaler_X.scale_,
            y_mean=self.bundle.scaler_y.mean_,
            y_scale=self.bundle.scaler_y.scale_,
        )
        meta = {
            "input_dim":    int(self.model._entry[0].in_features),
            "output_dim":   int(self.model._head.out_features),
            "feature_cols": list(self.bundle.metadata.get("feature_cols", ["x", "y", "vx", "vy"])),
            "target_cols":  list(self.bundle.metadata.get("target_cols",  ["ax", "ay"])),
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(path.parent / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        scalers_path = Path(path).parent / "scalers.npz"
        if scalers_path.exists():
            sc = np.load(scalers_path)
            self.bundle.scaler_X.mean_  = sc["X_mean"]
            self.bundle.scaler_X.scale_ = sc["X_scale"]
            self.bundle.scaler_y.mean_  = sc["y_mean"]
            self.bundle.scaler_y.scale_ = sc["y_scale"]
