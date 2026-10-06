"""Cross networks of DCNv2 (https://arxiv.org/abs/2008.13535)."""

import torch
from torch import Tensor, nn


class CrossNetV2(nn.Module):
    """Full-rank cross network: ``x_{l+1} = x_l + x_0 * (W_l x_l + b_l)``."""

    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))

    def forward(self, x0: Tensor) -> Tensor:
        x = x0
        for layer in self.layers:
            x = x + x0 * layer(x)
        return x


class CrossNetMix(nn.Module):
    """Mixture of low-rank experts cross network (DCN-Mix).

    Each expert maps ``x_l`` through ``tanh(V^T x)``, then ``tanh(C .)``, then ``U .`` (all
    low-rank), adds the bias and multiplies by ``x_0``. Experts are mixed with a softmax gate
    computed from ``x_l``, and the result is added to ``x_l``.

    Follows FuxiCTR v2.2.0: one gating layer is shared by all cross layers. Its initialization
    differs from FuxiCTR only in the Xavier fan-out (``num_experts`` instead of 1).
    """

    def __init__(self, dim: int, num_layers: int, low_rank: int = 32, num_experts: int = 4) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.u = nn.Parameter(torch.empty(num_layers, num_experts, dim, low_rank))
        self.v = nn.Parameter(torch.empty(num_layers, num_experts, dim, low_rank))
        self.c = nn.Parameter(torch.empty(num_layers, num_experts, low_rank, low_rank))
        self.bias = nn.Parameter(torch.zeros(num_layers, dim))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        with torch.no_grad():
            for param in (self.u, self.v, self.c):
                for layer_param in param:
                    nn.init.xavier_normal_(layer_param)

    def forward(self, x0: Tensor) -> Tensor:
        x = x0
        for i in range(self.num_layers):
            gate = torch.softmax(self.gate(x), dim=-1)  # (batch, experts)
            low = torch.tanh(torch.einsum("bd,edr->ber", x, self.v[i]))
            low = torch.tanh(torch.einsum("erk,bek->ber", self.c[i], low))
            experts = (torch.einsum("edr,ber->bed", self.u[i], low) + self.bias[i]) * x0.unsqueeze(
                1
            )
            x = torch.einsum("bed,be->bd", experts, gate) + x
        return x
