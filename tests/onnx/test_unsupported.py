"""ONNX conversion of passthrough pipelines is a later milestone: it must fail loudly, not guess."""

from pathlib import Path

import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from bars_dcn.estimator import DCNClassifier
from bars_dcn.onnx import to_onnx
from bars_dcn.preprocessing import OrdinalEncoder

SAMPLE = Path(__file__).parents[1] / "data" / "criteo_x4_sample"
CATS = [f"C{i}" for i in range(1, 27)]


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE / "train.parquet")


def _model(**kwargs) -> DCNClassifier:
    return DCNClassifier(
        embedding_dim=4,
        parallel_hidden_units=[8],
        max_epochs=1,
        batch_size=1024,
        device="cpu",
        random_state=0,
        **kwargs,
    )


def test_encoder_with_passthrough_columns_is_rejected(train):
    pipeline = Pipeline(
        [
            ("encode", OrdinalEncoder(columns=CATS[:5], min_count=3)),
            ("model", _model(cat_columns=CATS[:5])),
        ]
    )
    pipeline.fit(train.select(CATS), train["Label"])
    with pytest.raises(NotImplementedError, match="passthrough"):
        to_onnx(pipeline, CATS)


def test_integer_encoder_is_rejected(train):
    ints = train.select(CATS).with_columns((pl.all().hash() % 5).cast(pl.Int64))
    pipeline = Pipeline([("encode", OrdinalEncoder(min_count=3)), ("model", _model())])
    pipeline.fit(ints, train["Label"])
    with pytest.raises(NotImplementedError, match="integer"):
        to_onnx(pipeline, CATS)


def test_column_selection_in_the_estimator_is_rejected(train):
    pipeline = Pipeline(
        [("encode", OrdinalEncoder(min_count=3)), ("model", _model(cat_columns=CATS[:5]))]
    )
    pipeline.fit(train.select(CATS), train["Label"])
    with pytest.raises(NotImplementedError, match="cat_columns"):
        to_onnx(pipeline, CATS)
