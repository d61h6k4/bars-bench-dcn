"""PLE bins -> float numeric block -> DCN, exported to one ONNX graph, on the criteo sample."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
from skl2onnx import convert_sklearn
from sklearn.pipeline import Pipeline

from bars_dcn.estimator import DCNClassifier
from bars_dcn.onnx import TENSOR_TYPES, to_onnx
from bars_dcn.preprocessing import OrdinalEncoder, PiecewiseLinearEncoder, ple_name

SAMPLE = Path(__file__).parents[1] / "data" / "criteo_x4_sample"
NUMS = [f"I{i}" for i in range(1, 14)]
CATS = [f"C{i}" for i in range(1, 27)]
N_BINS = 6


def _frame(split: str) -> pl.DataFrame:
    return pl.read_parquet(SAMPLE / f"{split}.parquet")


@pytest.fixture(scope="module")
def fitted() -> tuple[Pipeline, PiecewiseLinearEncoder]:
    train = _frame("train")
    ple = PiecewiseLinearEncoder(columns=NUMS, n_bins=N_BINS)
    bins = ple.fit(train).transform(train.head(1)).columns[len(train.columns) :]
    pipeline = Pipeline(
        [
            ("ple", ple),
            ("encode", OrdinalEncoder(columns=CATS, min_count=3)),
            (
                "model",
                DCNClassifier(
                    cat_columns=CATS,
                    num_columns=bins,
                    embedding_dim=4,
                    num_cross_layers=2,
                    parallel_hidden_units=[16],
                    max_epochs=2,
                    batch_size=256,
                    device="cpu",
                    random_state=0,
                ),
            ),
        ]
    )
    pipeline.fit(train, train["Label"].to_numpy())
    return pipeline, ple


def test_onnx_matches_the_pipeline_with_missing_numerics(fitted):
    pipeline, _ = fitted
    requests = _frame("valid").select([*NUMS, *CATS]).head(300)
    requests = requests.with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0)
        .then(None)
        .otherwise(pl.col(NUMS[0]))
        .alias(NUMS[0])
    )
    inputs = {**dict.fromkeys(NUMS, "double"), **dict.fromkeys(CATS, "string")}
    session = ort.InferenceSession(to_onnx(pipeline, inputs).SerializeToString())
    feed = {
        name: np.array(
            [
                np.nan if v is None and name in NUMS else ("" if v is None else v)
                for v in requests[name]
            ],
            dtype=np.float64 if name in NUMS else object,
        ).reshape(-1, 1)
        for name in inputs
    }
    label, probabilities = session.run(None, feed)
    np.testing.assert_allclose(
        np.asarray(probabilities), pipeline.predict_proba(requests), atol=1e-5
    )
    np.testing.assert_array_equal(label, pipeline.predict(requests))


def test_ple_bins_are_exact_in_onnx(fitted):
    """The bin columns computed by the ONNX nodes equal polars' float32 values."""
    _, ple = fitted
    values = pl.DataFrame({"I1": [None, -3.0, 0.0, 1.0, 7.5, 1e6]}).with_columns(
        pl.col("I1").cast(pl.Float64)
    )
    ple_one = PiecewiseLinearEncoder(columns=["I1"], n_bins=N_BINS).fit(_frame("train"))
    expected = ple_one.transform(values).select(
        [ple_name("I1", k) for k in range(ple_one.n_bins_[0])]
    )
    assert ple.n_bins_[0] == ple_one.n_bins_[0]
    model = convert_sklearn(
        Pipeline([("ple", ple_one)]),
        initial_types=[("I1", TENSOR_TYPES["double"]([None, 1]))],
        target_opset={"": 18, "ai.onnx.ml": 3},
    )
    session = ort.InferenceSession(model.SerializeToString())
    got = session.run(None, {"I1": values["I1"].fill_null(np.nan).to_numpy().reshape(-1, 1)})
    bins = np.hstack(
        [np.asarray(part) for part in got[1:]]
    )  # the first output is the passthrough I1
    np.testing.assert_array_equal(bins, expected.to_numpy())
