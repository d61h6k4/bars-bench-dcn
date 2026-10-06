import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from bars_dcn.diagnostics import SignalPropagation, probe
from bars_dcn.estimator import DCNClassifier
from bars_dcn.model import DCNv2


def _model_and_batch():
    torch.manual_seed(0)
    model = DCNv2([5, 6, 7], embedding_dim=4, structure="parallel", num_cross_layers=2,
                  parallel_hidden_units=[8, 8], batch_norm=True)  # fmt: skip
    torch.nn.init.normal_(model.embedding.weight, std=0.5)  # non-degenerate signal
    x = torch.stack([torch.randint(0, c, (64,)) for c in (5, 6, 7)], dim=1)
    return model, x, torch.randint(0, 2, (64,)).float()


def _events(path):
    accumulator = EventAccumulator(str(path), size_guidance={"histograms": 0, "scalars": 0})
    accumulator.Reload()
    return accumulator


def _record(model, x, y):
    with SignalPropagation(model) as spp:
        functional.binary_cross_entropy_with_logits(model(x), y).backward()
    return spp


def test_points_follow_execution_order_with_branch_depths():
    model, x, y = _model_and_batch()
    spp = _record(model, x, y)
    assert [(p.branch, p.depth, p.label) for p in spp.points] == [
        ("embedding", 0, "embedding"),
        ("cross", 0, "x0"), ("cross", 1, "x1"), ("cross", 2, "x2"),
        ("parallel", 0, "parallel.relu1"), ("parallel", 1, "parallel.relu2"),
        ("head", 0, "logit"),
    ]  # fmt: skip
    assert all(p.backward is not None for p in spp.points)


def test_cross_points_are_the_exact_residual_stream():
    model, x, y = _model_and_batch()
    spp = _record(model, x, y)
    stream = {p.label: p.forward for p in spp.points if p.branch == "cross"}
    with torch.no_grad():
        x0 = model.embedding(x + model.offsets).flatten(1)
        x1 = x0 + x0 * model.cross.layers[0](x0)
        x2 = x1 + x0 * model.cross.layers[1](x1)
    torch.testing.assert_close(stream["x0"], x0)
    torch.testing.assert_close(stream["x1"], x1)
    torch.testing.assert_close(stream["x2"], x2)  # x_L is the cross network's own output


def test_deep_points_are_post_relu_and_head_is_the_logit_with_its_exact_gradient():
    model, x, y = _model_and_batch()
    spp = _record(model, x, y)
    deep = [p for p in spp.points if p.branch == "parallel"]
    assert all((p.forward >= 0).all() for p in deep)
    logits = model(x).detach()
    head = spp.points[-1]
    torch.testing.assert_close(head.forward.flatten(), logits)
    torch.testing.assert_close(head.backward.flatten(), (torch.sigmoid(logits) - y) / len(y))


def test_hooks_are_removed_and_probe_leaves_the_model_untouched():
    model, x, y = _model_and_batch()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    with SignalPropagation(model):
        pass
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())  # noqa: SLF001

    class Sink:  # probe must work with any SummaryWriter-like object
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    probe(model, x, y, Sink())  # ty: ignore[invalid-argument-type]
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():  # BatchNorm running stats untouched too
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"structure": "stacked", "stacked_hidden_units": [8]},
        {
            "structure": "stacked_parallel",
            "stacked_hidden_units": [8],
            "parallel_hidden_units": [8],
        },
        {"structure": "crossnet_only"},
        {
            "structure": "parallel",
            "parallel_hidden_units": [8],
            "use_low_rank_mixture": True,
            "low_rank": 2,
        },
    ],
)
def test_every_structure_can_be_probed(kwargs, tmp_path):
    model = DCNv2([5, 6], embedding_dim=4, **kwargs)
    torch.nn.init.normal_(model.embedding.weight, std=0.5)
    x = torch.randint(0, 5, (32, 2))
    y = torch.randint(0, 2, (32,)).float()
    with SummaryWriter(tmp_path) as writer:
        probe(model, x, y, writer)
    events = _events(tmp_path)
    assert events.Scalars("Signal_Prop_Forward/head/Layer_Std")
    assert events.Scalars("Signal_Prop_Backward/cross/Layer_Std")


def test_probe_writes_global_and_per_branch_tags_with_the_right_steps(tmp_path):
    model, x, y = _model_and_batch()
    with SummaryWriter(tmp_path) as writer:
        probe(model, x, y, writer)
    events = _events(tmp_path)
    for direction in ("Forward", "Backward"):
        prefix = f"Signal_Prop_{direction}"
        assert [s.step for s in events.Scalars(f"{prefix}/Layer_Std")] == list(range(7))
        assert [h.step for h in events.Histograms(f"{prefix}/Layer_Dist")] == list(range(7))
        assert [s.step for s in events.Scalars(f"{prefix}/cross/Layer_Std")] == [0, 1, 2]
        assert [s.step for s in events.Scalars(f"{prefix}/parallel/Layer_Std")] == [0, 1]
        assert [s.step for s in events.Scalars(f"{prefix}/head/Layer_Std")] == [0]
    zeros = events.Scalars("Signal_Prop_Forward/parallel/Zero_Fraction")
    assert [s.step for s in zeros] == [0, 1]
    assert all(0 <= s.value < 1 for s in zeros)
    spp = _record(model.eval(), x, y)
    np.testing.assert_allclose(
        [s.value for s in events.Scalars("Signal_Prop_Forward/Layer_Std")],
        [p.forward.std().item() for p in spp.points],
        rtol=1e-5,
    )


def test_fit_with_log_dir_writes_epoch_metrics_and_one_spp_run_per_epoch(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=(200, 3))
    y = (x[:, 0] > 2).astype(int)
    model = DCNClassifier(
        embedding_dim=2, parallel_hidden_units=[4], batch_size=64, max_epochs=2,
        device="cpu", log_dir=str(tmp_path), random_state=0,
    )  # fmt: skip
    model.fit(x, y, eval_set=(x, y))
    metrics = _events(tmp_path / "metrics")
    assert [s.step for s in metrics.Scalars("epoch/val_auc")] == [1, 2]
    assert metrics.Scalars("epoch/train_loss")[0].value == pytest.approx(
        model.history_[0]["train_loss"], rel=1e-5
    )
    assert sorted(p.name for p in (tmp_path / "spp").iterdir()) == [
        "epoch_000",
        "epoch_001",
        "epoch_002",
    ]
    assert _events(tmp_path / "spp" / "epoch_002").Scalars("Signal_Prop_Backward/Layer_Std")


def test_epoch_zero_probe_sees_the_initial_state_and_logging_does_not_change_training(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=(200, 3))
    y = (x[:, 0] > 2).astype(int)

    def fit(log_dir):
        return DCNClassifier(
            embedding_dim=2, parallel_hidden_units=[4], batch_size=64, max_epochs=2,
            device="cpu", log_dir=log_dir, random_state=0,
        ).fit(x, y, eval_set=(x, y))  # fmt: skip

    logged, plain = fit(str(tmp_path)), fit(None)
    assert logged.history_ == plain.history_  # probing consumes no randomness, changes no weights
    initial = _events(tmp_path / "spp" / "epoch_000").Scalars(
        "Signal_Prop_Forward/embedding/Layer_Std"
    )
    assert initial[0].value == pytest.approx(1e-4, rel=0.5)  # FuxiCTR embedding init N(0, 1e-4)
