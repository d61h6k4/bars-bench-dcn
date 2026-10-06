"""MLP block used by the deep part of DCNv2."""

from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence


class MLPBlock(nn.Module):
    """``Linear -> [BatchNorm] -> ReLU -> [Dropout]`` per hidden layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        *,
        batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(in_dim, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = units
        self.layers = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
