"""Fitted sklearn pipeline -> single ONNX graph -> onnxruntime, on raw string columns."""

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
import torch
from sklearn.pipeline import Pipeline
from torch import nn

from bars_dcn.estimator import DCNClassifier
from bars_dcn.onnx import to_onnx
from bars_dcn.preprocessing import OrdinalEncoder

if TYPE_CHECKING:
    from bars_dcn.model import DCNv2
    from bars_dcn.model.dcn import Structure

SAMPLE = Path(__file__).parents[1] / "data" / "criteo_x4_sample"
CATS = [f"C{i}" for i in range(1, 27)]
STRUCTURES: tuple[Structure, ...] = ("crossnet_only", "stacked", "parallel", "stacked_parallel")


def _randomize(model: DCNv2) -> None:
    """Untrained weights give ~0.5 everywhere; make outputs vary so parity is meaningful."""
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        model.embedding.weight.normal_(0, 0.3, generator=gen)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                assert module.running_mean is not None
                assert module.running_var is not None
                module.running_mean.normal_(0, 0.5, generator=gen)
                module.running_var.uniform_(0.5, 2.0, generator=gen)
                module.weight.uniform_(0.5, 1.5, generator=gen)
                module.bias.normal_(0, 0.2, generator=gen)


def _feed(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """What a serving process builds from a request: plain arrays, missing values as ''."""
    return {
        column: np.array(
            ["" if value is None else value for value in frame[column].to_list()], dtype=object
        ).reshape(-1, 1)
        for column in CATS
    }


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE / "train.parquet")


@pytest.fixture(scope="module")
def requests() -> pl.DataFrame:
    """Valid rows (real nulls and unseen values) plus all-null, all-empty and unseen-token rows."""
    valid = pl.read_parquet(SAMPLE / "valid.parquet").select(CATS).head(500)
    edge = pl.DataFrame(
        {column: [None, "", "zzz-unseen-token"] for column in CATS},
        schema=dict.fromkeys(CATS, pl.String),
    )
    return pl.concat([valid, edge])


def _fit(train: pl.DataFrame, structure: Structure, *, mixture: bool) -> Pipeline:
    pipeline = Pipeline(
        [
            ("encode", OrdinalEncoder(min_count=3)),
            (
                "model",
                DCNClassifier(
                    embedding_dim=8,
                    structure=structure,
                    num_cross_layers=2,
                    use_low_rank_mixture=mixture,
                    low_rank=4,
                    num_experts=3,
                    stacked_hidden_units=[24, 12],
                    parallel_hidden_units=[32, 16],
                    batch_norm=True,
                    dropout=0.1,
                    max_epochs=1,
                    batch_size=2048,
                    device="cpu",
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(train.select(CATS), train["Label"])
    _randomize(pipeline["model"].model_)
    return pipeline


@pytest.fixture(scope="module")
def fitted(train):
    return _fit(train, "stacked_parallel", mixture=False)


@pytest.fixture(scope="module")
def session(fitted):
    return ort.InferenceSession(to_onnx(fitted, dict.fromkeys(CATS, "string")).SerializeToString())


def test_onnx_matches_pipeline_on_real_and_edge_rows(fitted, session, requests):
    label, probabilities = session.run(None, _feed(requests))
    expected = fitted.predict_proba(requests)
    assert probabilities.shape == (len(requests), 2)
    np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(label, fitted.predict(requests))


def test_outputs_are_plain_tensors_not_zipmap(fitted):
    proto = to_onnx(fitted, dict.fromkeys(CATS, "string"))
    assert [output.name for output in proto.graph.output] == ["label", "probabilities"]
    assert "ZipMap" not in {node.op_type for node in proto.graph.node}


def test_predictions_are_not_degenerate(fitted, requests):
    positive = fitted.predict_proba(requests)[:, 1]
    assert positive.std() > 0.03
    assert 0.05 < positive.mean() < 0.95


def test_graph_responds_to_its_inputs(session, requests):
    feed = _feed(requests)
    original = session.run(None, feed)[1]
    changed = {**feed, "C3": np.array(["zzz"] * len(requests), dtype=object).reshape(-1, 1)}
    assert np.abs(session.run(None, changed)[1] - original).max() > 1e-3


def test_edge_rows_are_treated_as_oov_by_the_graph(fitted, session, requests):
    # all-null, all-empty and unseen-token rows must all give the all-OOV prediction
    probabilities = session.run(None, _feed(requests))[1]
    all_oov = probabilities[-3:, 1]
    np.testing.assert_allclose(all_oov, all_oov[0], atol=1e-6)
    assert not fitted["encode"].transform(requests.tail(3)).to_numpy().any()


@pytest.mark.parametrize("batch", [1, 7])
def test_dynamic_batch_size(fitted, session, requests, batch):
    part = requests.head(batch)
    probabilities = session.run(None, _feed(part))[1]
    np.testing.assert_allclose(probabilities, fitted.predict_proba(part), atol=1e-5)


@pytest.mark.parametrize(("structure", "mixture"), list(product(STRUCTURES, [False, True])))
def test_every_structure_and_cross_type_exports(train, requests, structure, mixture):
    pipeline = _fit(train, structure, mixture=mixture)
    session = ort.InferenceSession(
        to_onnx(pipeline, dict.fromkeys(CATS, "string")).SerializeToString()
    )
    probabilities = np.asarray(session.run(None, _feed(requests))[1])
    np.testing.assert_allclose(probabilities, pipeline.predict_proba(requests), rtol=0, atol=1e-5)
