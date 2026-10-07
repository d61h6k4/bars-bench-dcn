"""ScalarLens numerical embedding (Yao et al., arXiv 2609.29182). Pure torch, ONNX-exportable.

A numeric field gets a *stable coordinate* from its scalar alone (a piecewise-linear interpolation
between learned vectors over K learned monotone intervals of the training range) and a *contextual
response* (a few bounded low-rank updates over all field tokens). Only numeric tokens are returned;
the categorical embeddings are read, never rewritten.
"""

import torch
from torch import Tensor, nn

_EMBEDDING_STD = 1e-4  # as the categorical table (FuxiCTR init)
_MIN_GAP = 1e-12  # only matters when two float32 boundaries are exactly equal


class ScalarLens(nn.Module):
    """Map raw numerics ``(batch, N)`` and categorical embeddings ``(batch, C, d)`` to tokens.

    The output is one ``d``-dimensional numeric token per numeric field, ``(batch, N, d)``.

    ``set_ranges`` must be called with the training ``[low, high]`` of every field before use;
    inputs outside the range are clipped to it. Defaults are the paper's frozen ones
    (``K=16, T=3, r=16, m=8``, head width 32); ``temperature`` and ``norm_eps`` are not specified
    there and are assumed. ``token_dropout`` zeroes a whole numeric token with that probability
    while training (the survivors are rescaled), as scikit-rank's PLE ``feature_dropout``.
    """

    low: Tensor
    high: Tensor

    def __init__(
        self,
        n_numeric: int,
        n_categorical: int,
        dim: int = 16,
        *,
        intervals: int = 16,
        steps: int = 3,
        rank: int = 16,
        state: int = 8,
        head_width: int = 32,
        min_width: float = 1e-4,
        temperature: float = 1.0,
        norm_eps: float = 1e-6,
        token_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if intervals < 1:
            msg = f"intervals must be at least 1, got {intervals}"
            raise ValueError(msg)
        fields = n_numeric + n_categorical
        self.n_numeric, self.dim, self.intervals = n_numeric, dim, intervals
        self.steps, self.state = steps, state
        self.min_width, self.temperature, self.norm_eps = min_width, temperature, norm_eps
        self.token_dropout = token_dropout
        self.register_buffer("low", torch.zeros(n_numeric))
        self.register_buffer("high", torch.ones(n_numeric))
        self.width_logits = nn.Parameter(torch.zeros(n_numeric, intervals))
        self.knots = nn.Parameter(torch.randn(n_numeric, intervals + 1, dim) * _EMBEDDING_STD)
        # field-specific drive / gate heads: ebar_f (d) -> (delta_f, g_f) (m each)
        self.head_weight = nn.Parameter(torch.empty(fields, dim, 2 * state))
        self.head_bias = nn.Parameter(torch.zeros(fields, 2 * state))
        mixed = fields * state
        self.down = nn.Parameter(torch.randn(mixed, rank) / mixed**0.5)  # U
        self.up = nn.Parameter(torch.randn(rank, mixed) / rank**0.5)  # V
        self.step_logits = nn.Parameter(torch.zeros(steps))  # alpha_t = sigmoid(.)
        readout_in = dim + state * (1 + steps)
        # per-field readout MLP, stacked so one batched matmul serves every numeric field
        readouts = [
            (nn.Linear(readout_in, head_width), nn.Linear(head_width, dim))
            for _ in range(n_numeric)
        ]  # default Linear init, drawn field by field
        self.readout_w1 = nn.Parameter(
            torch.stack([first.weight.T for first, _ in readouts]).detach()
        )
        self.readout_b1 = nn.Parameter(torch.stack([first.bias for first, _ in readouts]).detach())
        self.readout_w2 = nn.Parameter(
            torch.stack([second.weight.T for _, second in readouts]).detach()
        )
        self.readout_b2 = nn.Parameter(
            torch.stack([second.bias for _, second in readouts]).detach()
        )
        self.gain = nn.Parameter(torch.zeros(n_numeric))  # positive gain = softplus(.)
        nn.init.xavier_normal_(self.head_weight)

    def set_ranges(self, low: Tensor, high: Tensor) -> None:
        """Store the training range of each numeric field (a constant field gets width 1)."""
        self.low.copy_(low)
        self.high.copy_(torch.where(high > low, high, low + 1))

    def init_boundaries(self, edges: Tensor) -> None:
        """Start the (still learnable) boundaries at ``edges``, an ``(N, K + 1)`` quantile grid.

        Equal widths over the raw range put almost every row of a heavy-tailed field in the
        first interval; quantile edges spread the rows over all intervals from the start.
        ``set_ranges`` must have been called first.
        """
        span = (self.high - self.low).unsqueeze(-1)
        fractions = edges.diff(dim=-1).to(span.dtype) / span
        k = self.intervals
        softmax = (fractions * (1 + k * self.min_width) - self.min_width).clamp_min(1e-6)
        with torch.no_grad():
            self.width_logits.copy_(self.temperature * softmax.log())

    def boundaries(self) -> Tensor:
        """Return the ``(N, K + 1)`` interval boundaries: strictly increasing from low to high."""
        weights = torch.softmax(self.width_logits / self.temperature, dim=-1)
        k = self.intervals
        widths = (weights + self.min_width) / (1 + k * self.min_width)
        span = (self.high - self.low).unsqueeze(-1)
        inner = self.low.unsqueeze(-1) + span * torch.cumsum(widths, dim=-1)
        return torch.cat([self.low.unsqueeze(-1), inner], dim=-1)

    def coordinate(self, x_num: Tensor) -> Tensor:
        """Return the stable coordinate ``(batch, N, d)``: a function of the scalar alone."""
        bounds = self.boundaries()
        x = torch.minimum(torch.maximum(x_num, self.low), self.high)
        # interval index = number of inner boundaries <= x, in [0, K - 1]
        index = (x.unsqueeze(-1) >= bounds[:, 1:-1]).sum(-1)
        slots = torch.arange(self.intervals + 1, device=x.device)
        lower = (index.unsqueeze(-1) == slots).to(x.dtype)  # (B, N, K + 1) one-hot of k
        upper = (index.unsqueeze(-1) + 1 == slots).to(x.dtype)  # one-hot of k + 1
        start, end = (lower * bounds).sum(-1), (upper * bounds).sum(-1)
        # float32 can collapse neighbouring boundaries (|low| >> span): then x == start, weight 0
        weight = ((x - start) / (end - start).clamp_min(_MIN_GAP)).unsqueeze(-1)
        return torch.einsum("bnk,nkd->bnd", lower, self.knots) * (
            1 - weight
        ) + weight * torch.einsum("bnk,nkd->bnd", upper, self.knots)

    def forward(self, x_num: Tensor, cat_embeddings: Tensor) -> Tensor:
        coordinate = self.coordinate(x_num)
        tokens = torch.cat([coordinate, cat_embeddings], dim=1)  # numeric fields first
        batch, fields, _ = tokens.shape
        mean_square = tokens.pow(2).mean(-1, keepdim=True)
        normalized = tokens * torch.rsqrt(mean_square + self.norm_eps)
        heads = torch.einsum("bfd,fdm->bfm", normalized, self.head_weight) + self.head_bias
        drive, gate = heads.chunk(2, dim=-1)  # (B, F, m) each
        drive, gate = drive.reshape(batch, -1), gate.reshape(batch, -1)
        state = torch.tanh(drive)
        states = []
        for t in range(self.steps):
            update = (state @ self.down) @ self.up
            proposal = torch.tanh(drive + gate * update)
            alpha = torch.sigmoid(self.step_logits[t])
            state = (1 - alpha) * state + alpha * proposal
            states.append(state.reshape(batch, fields, self.state)[:, : self.n_numeric])
        gate = gate.reshape(batch, fields, self.state)[:, : self.n_numeric]
        features = torch.cat([normalized[:, : self.n_numeric], gate, *states], dim=-1)
        hidden = torch.nn.functional.silu(
            torch.einsum("bnf,nfw->bnw", features, self.readout_w1) + self.readout_b1
        )
        out = torch.einsum("bnw,nwd->bnd", hidden, self.readout_w2) + self.readout_b2  # (B, N, d)
        out = out * torch.nn.functional.softplus(self.gain).unsqueeze(-1)
        if self.training and self.token_dropout > 0:
            keep = torch.rand(batch, self.n_numeric, 1, device=out.device) >= self.token_dropout
            out = out * keep.to(out.dtype) / (1 - self.token_dropout)
        return out
