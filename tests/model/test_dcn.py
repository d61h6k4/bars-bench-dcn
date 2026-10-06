import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from bars_dcn.model import DCNv2

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "data" / "bars_dcnv2_criteo_x4_001.json").read_text()
)
CARDS = [3, 4, 5]
EMB, NUM = 4, 2
DIM = len(CARDS) * EMB + NUM


def _batch(n: int = 6):
    gen = torch.Generator().manual_seed(0)
    x_cat = torch.stack([torch.randint(0, c, (n,), generator=gen) for c in CARDS], dim=1)
    return x_cat, torch.randn(n, NUM, generator=gen)


def _model(structure, **kwargs):
    return DCNv2(
        CARDS,
        NUM,
        embedding_dim=EMB,
        structure=structure,
        num_cross_layers=2,
        stacked_hidden_units=[10, 6],
        parallel_hidden_units=[8, 5],
        **kwargs,
    )


@pytest.mark.parametrize(
    ("structure", "final_dim"),
    [
        ("crossnet_only", DIM),
        ("stacked", 6),
        ("parallel", DIM + 5),
        ("stacked_parallel", 6 + 5),
    ],
)
@pytest.mark.parametrize("mixture", [False, True])
def test_structure_final_width_and_output_shape(structure, final_dim, mixture):
    model = _model(structure, use_low_rank_mixture=mixture, low_rank=3, num_experts=2)
    assert model.fc.in_features == final_dim
    assert model(*_batch()).shape == (6,)


def test_bars_reference_parameter_count():
    cfg = REFERENCE["model"]
    model = DCNv2(
        list(REFERENCE["vocab_sizes"].values()),
        embedding_dim=cfg["embedding_dim"],
        structure=cfg["model_structure"],
        num_cross_layers=cfg["num_cross_layers"],
        parallel_hidden_units=cfg["parallel_dnn_hidden_units"],
        batch_norm=cfg["batch_norm"],
        dropout=cfg["net_dropout"],
    )
    assert sum(p.numel() for p in model.parameters()) == REFERENCE["expected"]["total_parameters"]


def test_initialization_matches_fuxictr():
    model = DCNv2([2000] * 5, embedding_dim=16, parallel_hidden_units=[512, 512], batch_norm=True)
    assert model.embedding.weight.std().item() == pytest.approx(1e-4, rel=0.05)
    assert model.parallel is not None
    first = model.parallel.layers[0]
    assert isinstance(first, nn.Linear)
    dim = 5 * 16
    assert first.weight.std().item() == pytest.approx(math.sqrt(2 / (dim + 512)), rel=0.05)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert not module.bias.any()


def test_field_offsets_keep_fields_apart():
    model = DCNv2([3, 4], embedding_dim=2, structure="crossnet_only")
    model(torch.tensor([[1, 1]])).sum().backward()
    grad = model.embedding.weight.grad
    assert grad is not None
    rows_with_grad = grad.abs().sum(dim=1).nonzero().flatten().tolist()
    assert rows_with_grad == [1, 3 + 1]


def test_numeric_block_is_required_when_declared():
    model = _model("parallel")
    with pytest.raises(ValueError, match="x_num is required"):
        model(_batch()[0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"structure": "bogus"}, "not supported"),
        ({"structure": "stacked"}, "requires stacked_hidden_units"),
        ({"structure": "parallel"}, "requires parallel_hidden_units"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DCNv2(CARDS, **kwargs)


def test_eval_mode_is_deterministic():
    model = _model("stacked_parallel", batch_norm=True, dropout=0.5).eval()
    batch = _batch()
    torch.testing.assert_close(model(*batch), model(*batch))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("mixture", [False, True])
def test_cpu_and_mps_agree(mixture):
    model = _model(
        "stacked_parallel", batch_norm=True, use_low_rank_mixture=mixture, low_rank=3
    ).eval()
    batch = _batch()
    expected = model(*batch)
    model_mps = model.to("mps")
    actual = model_mps(*(t.to("mps") for t in batch)).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
