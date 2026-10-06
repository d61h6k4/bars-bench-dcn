"""The FuxiCTR-replica training loop (see bars_dcn.training)."""

from itertools import chain
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from bars_dcn.estimator import DCNClassifier
from bars_dcn.model import DCNv2
from bars_dcn.pipeline import fit_pipeline
from bars_dcn.preprocessing import LogSquaredBucketizer, OrdinalEncoder
from bars_dcn.training import PlateauStopper, embedding_penalty

SAMPLE = Path(__file__).parent / "data" / "criteo_x4_sample"
NUMS = [f"I{i}" for i in range(1, 14)]
CATS = [f"C{i}" for i in range(1, 27)]

# Validation AUC per epoch from the BARS DCNv2_criteo_x4_001_005 log (FuxiCTR v2.2.0)
BARS_VALID_AUC = [
    0.806450,
    0.809079,
    0.810513,
    0.811169,
    0.811757,
    0.811747,
    0.814037,
    0.813989,
    0.813659,
]


class TestPlateauStopper:
    def test_replays_the_bars_log_exactly(self):
        stopper = PlateauStopper()
        lr, improved, lrs, stopped_at = 1e-3, [], [], None
        for epoch, auc in enumerate(BARS_VALID_AUC, start=1):
            decision = stopper.update(auc, lr)
            lrs.append(decision.lr)
            if decision.improved:
                improved.append(epoch)
            lr = decision.lr
            if decision.stop:
                stopped_at = epoch
                break
        assert improved == [1, 2, 3, 4, 5, 7]  # epoch 7 is the best (0.814037)
        assert stopped_at == 9  # "Epoch==9 early stop" in the log
        # log: lr -> 1e-4 after epoch 6, 1e-5 after epoch 8, 1e-6 after epoch 9
        expected = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-6]
        np.testing.assert_allclose(lrs, expected)

    def test_improvement_needs_min_delta(self):
        stopper = PlateauStopper(min_delta=1e-6)
        assert stopper.update(0.8, 1e-3).improved
        assert not stopper.update(0.8 + 5e-7, 1e-3).improved  # below best + min_delta
        assert stopper.update(0.8 + 2e-6, 1e-3).improved

    def test_learning_rate_is_floored_at_min_lr(self):
        stopper = PlateauStopper(patience=100, min_lr=1e-6)
        stopper.update(0.9, 1e-3)
        lr = 1e-3
        for _ in range(10):
            lr = stopper.update(0.5, lr).lr
        assert lr == pytest.approx(1e-6)

    def test_improvement_resets_patience_but_not_the_learning_rate(self):
        stopper = PlateauStopper(patience=2)
        stopper.update(0.80, 1e-3)
        reduced = stopper.update(0.79, 1e-3)
        assert reduced.lr == pytest.approx(1e-4)
        assert not reduced.stop
        better = stopper.update(0.81, reduced.lr)
        assert better.improved
        assert better.lr == pytest.approx(1e-4)  # not restored
        assert not stopper.update(0.80, better.lr).stop  # patience counts from zero again


def test_embedding_penalty_is_half_lambda_squared_norm_with_gradient_lambda_w():
    model = DCNv2([3, 4], embedding_dim=2, structure="crossnet_only")
    torch.nn.init.normal_(model.embedding.weight)
    penalty = embedding_penalty(model, 1e-2)
    assert penalty.item() == pytest.approx(0.5 * 1e-2 * model.embedding.weight.pow(2).sum().item())
    penalty.backward()
    assert model.embedding.weight.grad is not None
    torch.testing.assert_close(model.embedding.weight.grad, 1e-2 * model.embedding.weight)


def _pipeline(max_epochs: int = 30) -> Pipeline:
    model = DCNClassifier(
        cat_columns=NUMS + CATS,
        embedding_dim=8,
        parallel_hidden_units=[64, 64],
        batch_size=512,
        max_epochs=max_epochs,
        device="cpu",
        random_state=0,
    )
    return Pipeline(
        [
            ("bucket", LogSquaredBucketizer(columns=NUMS)),
            ("num", OrdinalEncoder(columns=NUMS, min_count=3, na_values=[0])),
            ("cat", OrdinalEncoder(columns=CATS, min_count=3)),
            ("model", model),
        ]
    )


@pytest.fixture(scope="module")
def frames() -> dict[str, pl.DataFrame]:
    return {name: pl.read_parquet(SAMPLE / f"{name}.parquet") for name in ("train", "valid")}


@pytest.fixture(scope="module")
def trained(frames) -> Pipeline:
    train, valid = frames["train"], frames["valid"]
    return fit_pipeline(
        _pipeline(),
        train.drop("Label"),
        train["Label"],
        eval_set=(valid.drop("Label"), valid["Label"]),
    )


class TestFitWithValidation:
    def test_learns_something_real(self, trained):
        model = trained["model"]
        best = model.history_[model.best_epoch_ - 1]
        assert best["val_auc"] > 0.70  # untrained is ~0.5; observed ~0.727 on this sample
        assert model.history_[1]["train_loss"] < model.history_[0]["train_loss"]

    def test_history_has_one_record_per_epoch_with_the_expected_fields(self, trained):
        history = trained["model"].history_
        assert [h["epoch"] for h in history] == list(range(1, len(history) + 1))
        assert set(history[0]) == {"epoch", "lr", "train_loss", "val_auc", "val_logloss"}
        assert history[0]["lr"] == pytest.approx(1e-3)

    def test_schedule_and_stopping_follow_the_stopper(self, trained):
        # replay the recorded validation AUCs: learning rates, stop epoch and best epoch must agree
        history = trained["model"].history_
        stopper, lr, best_epoch = PlateauStopper(), 1e-3, None
        for record in history:
            assert record["lr"] == pytest.approx(lr)
            decision = stopper.update(record["val_auc"], lr)
            lr = decision.lr
            if decision.improved:
                best_epoch = record["epoch"]
        assert decision.stop
        assert trained["model"].best_epoch_ == best_epoch

    def test_best_epoch_weights_are_restored(self, trained, frames):
        model = trained["model"]
        assert model.best_epoch_ < len(model.history_)  # otherwise this test proves nothing
        valid = frames["valid"]
        probabilities = trained.predict_proba(valid.drop("Label"))[:, 1]
        auc = roc_auc_score(valid["Label"], probabilities)
        assert auc == pytest.approx(model.history_[model.best_epoch_ - 1]["val_auc"], abs=1e-9)

    def test_training_is_reproducible_on_cpu(self, trained, frames):
        train, valid = frames["train"], frames["valid"]
        again = fit_pipeline(
            _pipeline(),
            train.drop("Label"),
            train["Label"],
            eval_set=(valid.drop("Label"), valid["Label"]),
        )
        assert again["model"].history_ == trained["model"].history_

    def test_model_ends_on_the_cpu(self, trained):
        assert {p.device.type for p in trained["model"].model_.parameters()} == {"cpu"}


class TestFitWithoutValidation:
    def test_trains_exactly_max_epochs_at_a_fixed_rate(self, frames):
        train = frames["train"]
        pipeline = _pipeline(max_epochs=3)
        fit_pipeline(pipeline, train.drop("Label"), train["Label"])
        history = pipeline["model"].history_
        assert len(history) == 3
        assert all(h["lr"] == pytest.approx(1e-3) for h in history)
        assert all("val_auc" not in h for h in history)
        assert pipeline["model"].best_epoch_ is None


def _toy(n: int = 41):
    rng = np.random.default_rng(0)
    x = rng.integers(0, [4, 5], size=(n, 2))
    return x, (x[:, 0] % 2).astype(int)


class TestEdgeCases:
    def test_a_trailing_single_row_batch_does_not_break_batch_norm(self):
        x, y = _toy(33)  # batch_size 16 -> batches of 16, 16, 1
        model = DCNClassifier(
            embedding_dim=2, parallel_hidden_units=[4], batch_norm=True,
            batch_size=16, max_epochs=1, device="cpu", random_state=0,
        )  # fmt: skip
        model.fit(x, y)
        assert len(model.history_) == 1

    def test_unsupported_device_is_rejected(self):
        x, y = _toy()
        with pytest.raises(ValueError, match="device must be"):
            DCNClassifier(parallel_hidden_units=[4], max_epochs=1, device="cuda").fit(x, y)

    def test_eval_labels_must_have_been_seen_in_fit(self):
        x, y = _toy()
        model = DCNClassifier(parallel_hidden_units=[4], max_epochs=1, device="cpu")
        with pytest.raises(ValueError, match="not seen in fit"):
            model.fit(x, y, eval_set=(x, np.full(len(y), 7)))

    def test_eval_indices_beyond_the_training_cardinality_are_rejected(self):
        x, y = _toy()
        bad = x.copy()
        bad[0, 0] = x[:, 0].max() + 1
        model = DCNClassifier(parallel_hidden_units=[4], max_epochs=1, device="cpu")
        with pytest.raises(ValueError, match="out of range"):
            model.fit(x, y, eval_set=(bad, y))

    def test_gradient_clipping_is_applied_on_every_step_with_the_configured_norm(self, monkeypatch):
        calls = []
        original = Accelerator.clip_grad_norm_

        def spy(self, parameters, max_norm, *args, **kwargs):
            calls.append(max_norm)
            return original(self, parameters, max_norm, *args, **kwargs)

        monkeypatch.setattr(Accelerator, "clip_grad_norm_", spy)
        x, y = _toy(40)
        DCNClassifier(
            embedding_dim=2, parallel_hidden_units=[4], batch_size=16, max_epochs=2,
            max_grad_norm=3.5, device="cpu", random_state=0,
        ).fit(x, y)  # fmt: skip
        assert calls == [3.5] * 6  # 3 batches (16, 16, 8) x 2 epochs


def test_rows_are_reshuffled_every_epoch(monkeypatch):
    seen: list[list[int]] = []
    original = DCNv2.forward

    def record(self, index, *args, **kwargs):
        if self.training:
            seen.append(index[:, 0].tolist())
        return original(self, index, *args, **kwargs)

    monkeypatch.setattr(DCNv2, "forward", record)
    x = np.arange(32).reshape(-1, 1)
    y = x[:, 0] % 2
    DCNClassifier(
        embedding_dim=2, parallel_hidden_units=[4], batch_size=8, max_epochs=2,
        device="cpu", random_state=0,
    ).fit(x, y)  # fmt: skip
    first, second = list(chain(*seen[:4])), list(chain(*seen[4:]))
    assert sorted(first) == sorted(second) == list(range(32))  # every row once per epoch
    assert first != list(range(32))
    assert first != second


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_device_trains_on_mps_and_returns_a_cpu_model(monkeypatch):
    seen = set()
    original = DCNv2.forward

    def record(self, index, *args, **kwargs):
        seen.add(index.device.type)
        return original(self, index, *args, **kwargs)

    monkeypatch.setattr(DCNv2, "forward", record)
    x, y = _toy()
    model = DCNClassifier(parallel_hidden_units=[4], embedding_dim=2, max_epochs=1, device="mps")
    model.fit(x, y)
    assert seen == {"mps"}
    assert {p.device.type for p in model.model_.parameters()} == {"cpu"}
