import pytest
import torch
from torch import nn

from bars_dcn.model import CrossNetMix, CrossNetV2

BATCH, DIM = 7, 12


@pytest.fixture
def x0():
    return torch.randn(BATCH, DIM, generator=torch.Generator().manual_seed(0))


def _linears(net: CrossNetV2) -> list[nn.Linear]:
    layers = [layer for layer in net.layers if isinstance(layer, nn.Linear)]
    assert len(layers) == len(net.layers)
    return layers


def test_cross_net_v2_matches_naive_formula(x0):
    net = CrossNetV2(DIM, num_layers=3)
    for layer in _linears(net):  # non-trivial biases, init is zero
        torch.nn.init.normal_(layer.bias)
    expected = x0
    for layer in _linears(net):
        expected = expected + x0 * (expected @ layer.weight.T + layer.bias)
    torch.testing.assert_close(net(x0), expected)


def test_cross_net_v2_zero_weights_is_identity(x0):
    net = CrossNetV2(DIM, num_layers=2)
    for layer in _linears(net):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    torch.testing.assert_close(net(x0), x0)


def _mix_reference(net: CrossNetMix, inputs: torch.Tensor) -> torch.Tensor:
    """Per-expert loop, written after the FuxiCTR v2.2.0 CrossNetMix (column-vector form)."""
    num_experts = net.gate.weight.shape[0]
    x_0 = inputs.unsqueeze(2)  # (batch, dim, 1)
    x_l = x_0
    for i in range(net.num_layers):
        outputs, scores = [], []
        for e in range(num_experts):
            scores.append(x_l.squeeze(2) @ net.gate.weight[e].unsqueeze(1))  # (batch, 1)
            v_x = torch.tanh(net.v[i, e].T @ x_l)  # (batch, rank, 1)
            v_x = torch.tanh(net.c[i, e] @ v_x)
            uv_x = net.u[i, e] @ v_x  # (batch, dim, 1)
            outputs.append((x_0 * (uv_x + net.bias[i].unsqueeze(1))).squeeze(2))
        outputs = torch.stack(outputs, dim=2)  # (batch, dim, experts)
        scores = torch.stack(scores, dim=1)  # (batch, experts, 1)
        x_l = outputs @ scores.softmax(dim=1) + x_l
    return x_l.squeeze(2)


def test_cross_net_mix_matches_reference_loop(x0):
    net = CrossNetMix(DIM, num_layers=3, low_rank=5, num_experts=4)
    torch.nn.init.normal_(net.bias)  # non-trivial biases, init is zero
    torch.testing.assert_close(net(x0), _mix_reference(net, x0), rtol=1e-5, atol=1e-5)
