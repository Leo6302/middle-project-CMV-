from __future__ import annotations
import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Residual block: Linear→BN→Act→Linear→BN, skip-connection + Act."""
    def __init__(self, dim: int, Act: type) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            Act(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = Act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class PhysicsMLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: list[int] = None,
        output_dim: int = 2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 128]

        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
        Act = act_map.get(activation, nn.SiLU)

        # Entry projection: input → first hidden dim
        self._entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            Act(),
        )

        # Hidden blocks: residual when adjacent dims match, plain otherwise
        self._blocks = nn.ModuleList()
        prev = hidden_dims[0]
        for h in hidden_dims[1:]:
            if h == prev:
                self._blocks.append(_ResBlock(h, Act))
            else:
                self._blocks.append(nn.Sequential(
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    Act(),
                ))
            prev = h

        self._head = nn.Linear(prev, output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._entry(x)
        for block in self._blocks:
            x = block(x)
        return self._head(x)
