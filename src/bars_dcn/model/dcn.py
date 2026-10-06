"""DCNv2 network. Pure torch: no sklearn or polars imports, so it exports to ONNX on its own."""

from itertools import accumulate
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

from bars_dcn.model.cross import CrossNetMix, CrossNetV2
from bars_dcn.model.mlp import MLPBlock

if TYPE_CHECKING:
    from collections.abc import Sequence

Structure = Literal["crossnet_only", "stacked", "parallel", "stacked_parallel"]
_STRUCTURES = ("crossnet_only", "stacked", "parallel", "stacked_parallel")
_EMBEDDING_STD = 1e-4


class DCNv2(nn.Module):
    """Deep & Cross Network v2 over embedded categorical fields and an optional numeric block.

    Inputs are integer indices (one column per categorical field, local to the field, i.e.
    ``0 <= x_cat[:, f] < cat_cardinalities[f]``) and an optional float block. All fields share
    one embedding table; per-field offsets keep them apart. With ``shared_embedding`` there are
    no offsets: every position indexes the same ``max(cat_cardinalities)``-row table (hashed ids).
    The output is the **logit**, shape ``(batch,)``; apply a sigmoid for probabilities.

    ``structure``: ``crossnet_only`` (cross output feeds the final layer), ``stacked`` (cross
    output goes through the stacked MLP), ``parallel`` (cross and MLP outputs are concatenated)
    or ``stacked_parallel`` (both MLPs, outputs concatenated).
    """

    offsets: Tensor

    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        num_features: int = 0,
        *,
        embedding_dim: int = 16,
        structure: Structure = "parallel",
        num_cross_layers: int = 3,
        use_low_rank_mixture: bool = False,
        low_rank: int = 32,
        num_experts: int = 4,
        stacked_hidden_units: Sequence[int] | None = None,
        parallel_hidden_units: Sequence[int] | None = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
        shared_embedding: bool = False,
    ) -> None:
        super().__init__()
        if structure not in _STRUCTURES:
            msg = f"structure={structure!r} not supported, expected one of {_STRUCTURES}"
            raise ValueError(msg)
        self.num_features = num_features
        if shared_embedding:
            rows = max(cat_cardinalities)
            offsets = [0] * len(cat_cardinalities)
        else:
            rows = sum(cat_cardinalities)
            offsets = [0, *accumulate(cat_cardinalities)][:-1]
        self.embedding = nn.Embedding(rows, embedding_dim)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        dim = len(cat_cardinalities) * embedding_dim + num_features

        if use_low_rank_mixture:
            self.cross = CrossNetMix(dim, num_cross_layers, low_rank, num_experts)
        else:
            self.cross = CrossNetV2(dim, num_cross_layers)

        self.stacked: MLPBlock | None = None
        self.parallel: MLPBlock | None = None
        if structure in ("stacked", "stacked_parallel"):
            if not stacked_hidden_units:
                msg = f"structure={structure!r} requires stacked_hidden_units"
                raise ValueError(msg)
            self.stacked = MLPBlock(
                dim, stacked_hidden_units, batch_norm=batch_norm, dropout=dropout
            )
        if structure in ("parallel", "stacked_parallel"):
            if not parallel_hidden_units:
                msg = f"structure={structure!r} requires parallel_hidden_units"
                raise ValueError(msg)
            self.parallel = MLPBlock(
                dim, parallel_hidden_units, batch_norm=batch_norm, dropout=dropout
            )

        # The cross output (or the stacked MLP applied to it) is always part of the final input;
        # the parallel MLP output is appended when present.
        final_dim = self.stacked.output_dim if self.stacked else dim
        if self.parallel:
            final_dim += self.parallel.output_dim
        self.fc = nn.Linear(final_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Embeddings ~ N(0, 1e-4); every Linear Xavier-normal with zero bias (as FuxiCTR)."""
        nn.init.normal_(self.embedding.weight, std=_EMBEDDING_STD)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_cat: Tensor, x_num: Tensor | None = None) -> Tensor:
        features = self.embedding(x_cat + self.offsets).flatten(1)
        if self.num_features:
            if x_num is None:
                msg = f"model was built with num_features={self.num_features}, x_num is required"
                raise ValueError(msg)
            features = torch.cat([features, x_num], dim=1)

        out = self.cross(features)
        if self.stacked is not None:
            out = self.stacked(out)
        if self.parallel is not None:
            out = torch.cat([out, self.parallel(features)], dim=-1)
        return self.fc(out).squeeze(-1)
