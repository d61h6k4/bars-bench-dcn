"""Raw numerics -> ScalarLens inside the model, exported to one ONNX graph, on the criteo sample."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from bars_dcn.estimator import DCNClassifier
from bars_dcn.onnx import to_onnx
from bars_dcn.preprocessing import MissingFiller, OrdinalEncoder

SAMPLE = Path(__file__).parents[1] / "data" / "criteo_x4_sample"
NUMS = [f"I{i}" for i in range(1, 14)]
CATS = [f"C{i}" for i in range(1, 27)]


@pytest.fixture(scope="module")
def fitted() -> Pipeline:
    train = pl.read_parquet(SAMPLE / "train.parquet")
    pipeline = Pipeline(
        [
            ("fill", MissingFiller(columns=NUMS, fill_value=0)),
            ("encode", OrdinalEncoder(columns=CATS, min_count=3)),
            (
                "model",
                DCNClassifier(
                    cat_columns=CATS,
                    num_columns=NUMS,
                    scalarlens=True,
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
    return pipeline.fit(train, train["Label"].to_numpy())


def test_the_ranges_are_recorded_from_the_training_numerics(fitted):
    train = pl.read_parquet(SAMPLE / "train.parquet").select(NUMS).fill_null(0)
    lens = fitted["model"].model_.numeric_embedding
    np.testing.assert_allclose(lens.low.numpy(), train.min().to_numpy().ravel())
    expected_high = train.max().to_numpy().ravel()
    np.testing.assert_allclose(lens.high.numpy(), np.maximum(expected_high, lens.low.numpy() + 1))


def test_onnx_matches_the_pipeline_with_missing_and_out_of_range_numerics(fitted):
    requests = pl.read_parquet(SAMPLE / "valid.parquet").select([*NUMS, *CATS]).head(300)
    requests = requests.with_columns(
        pl.when(pl.int_range(pl.len()) % 5 == 0).then(None).otherwise(pl.col("I1")).alias("I1"),
        pl.when(pl.int_range(pl.len()) % 11 == 0).then(10**9).otherwise(pl.col("I2")).alias("I2"),
    )
    inputs = {**dict.fromkeys(NUMS, "double"), **dict.fromkeys(CATS, "string")}
    session = ort.InferenceSession(to_onnx(fitted, inputs).SerializeToString())
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
    np.testing.assert_allclose(np.asarray(probabilities), fitted.predict_proba(requests), atol=1e-5)
    np.testing.assert_array_equal(label, fitted.predict(requests))
