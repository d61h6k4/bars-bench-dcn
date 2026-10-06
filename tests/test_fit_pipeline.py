from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from bars_dcn.estimator import DCNClassifier
from bars_dcn.pipeline import fit_pipeline
from bars_dcn.preprocessing import OrdinalEncoder

SAMPLE = Path(__file__).parent / "data" / "criteo_x4_sample"
CATS = [f"C{i}" for i in range(1, 27)]


def _pipeline(*, steps: bool = True) -> Pipeline:
    model = DCNClassifier(
        embedding_dim=4, parallel_hidden_units=[16], batch_size=1024, max_epochs=3,
        device="cpu", random_state=0,
    )  # fmt: skip
    return Pipeline(
        [("encode", OrdinalEncoder(min_count=3)), ("model", model)] if steps else [("model", model)]
    )


@pytest.fixture(scope="module")
def data():
    train = pl.read_parquet(SAMPLE / "train.parquet")
    valid = pl.read_parquet(SAMPLE / "valid.parquet")
    return train.select(CATS), train["Label"], valid.select(CATS), valid["Label"]


def test_matches_fitting_the_steps_by_hand(data):
    x, y, xv, yv = data
    pipeline = fit_pipeline(_pipeline(), x, y, eval_set=(xv, yv))

    encoder = OrdinalEncoder(min_count=3).fit(x)
    manual = _pipeline()["model"].fit(
        encoder.transform(x), y, eval_set=(encoder.transform(xv), yv.to_numpy())
    )
    np.testing.assert_array_equal(
        pipeline.predict_proba(xv), manual.predict_proba(encoder.transform(xv))
    )
    assert pipeline["model"].history_ == manual.history_


def test_preprocessing_is_fitted_on_training_data_only(data):
    x, y, xv, yv = data
    only_in_valid = xv.with_columns(pl.lit("zzz-valid-only").alias("C1"))
    pipeline = fit_pipeline(_pipeline(), x, y, eval_set=(only_in_valid, yv))
    assert "zzz-valid-only" not in pipeline["encode"].categories_[0]


def test_eager_and_lazy_inputs_give_the_same_model(data):
    x, y, xv, yv = data
    eager = fit_pipeline(_pipeline(), x, y, eval_set=(xv, yv))
    lazy = fit_pipeline(_pipeline(), x.lazy(), y, eval_set=(xv.lazy(), yv))
    assert lazy["model"].history_ == eager["model"].history_


def test_returns_the_same_pipeline_object_still_a_plain_sklearn_pipeline(data):
    x, y, xv, yv = data
    pipeline = _pipeline()
    assert fit_pipeline(pipeline, x, y, eval_set=(xv, yv)) is pipeline
    assert isinstance(pipeline, Pipeline)
    assert pipeline.predict_proba(xv).shape == (len(xv), 2)


def test_works_without_an_eval_set_and_skips_passthrough_steps(data):
    x, y, xv, _ = data
    pipeline = Pipeline([("skip", "passthrough"), *_pipeline().steps])
    fit_pipeline(pipeline, x, y)
    assert pipeline["model"].best_epoch_ is None
    assert pipeline.predict_proba(xv).shape == (len(xv), 2)


def test_estimator_only_pipeline(data):
    x, y, xv, yv = data
    encoded_train = OrdinalEncoder(min_count=3).fit(x)
    pipeline = _pipeline(steps=False)
    fit_pipeline(
        pipeline, encoded_train.transform(x), y, eval_set=(encoded_train.transform(xv), yv)
    )
    assert len(pipeline["model"].history_) >= 1
