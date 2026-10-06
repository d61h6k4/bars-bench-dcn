"""The BARS-style preprocessing declared as passthrough steps in a plain sklearn Pipeline."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from bars_dcn.estimator import DCNClassifier
from bars_dcn.preprocessing import LogSquaredBucketizer, OrdinalEncoder

SAMPLE = Path(__file__).parent / "data" / "criteo_x4_sample"
NUMS = [f"I{i}" for i in range(1, 14)]
CATS = [f"C{i}" for i in range(1, 27)]


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("bucket", LogSquaredBucketizer(columns=NUMS)),
            ("encode_numeric", OrdinalEncoder(columns=NUMS, min_count=3, na_values=[0])),
            ("encode_categorical", OrdinalEncoder(columns=CATS, min_count=3)),
            (
                "model",
                DCNClassifier(
                    cat_columns=NUMS + CATS,
                    embedding_dim=4,
                    parallel_hidden_units=[16],
                    max_epochs=1,
                    batch_size=2048,
                    device="cpu",
                    random_state=0,
                ),
            ),
        ]
    )


@pytest.fixture(scope="module")
def fitted() -> Pipeline:
    train = pl.scan_parquet(SAMPLE / "train.parquet")
    labels = train.select("Label").collect().to_series().to_numpy()
    return _pipeline().fit(train.drop("Label"), labels)  # fit on a LazyFrame


@pytest.fixture(scope="module")
def valid() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE / "valid.parquet").drop("Label")


def test_model_sees_exactly_the_declared_fields_in_order(fitted):
    model = fitted["model"]
    assert model.n_features_in_ == len(NUMS) + len(CATS)
    expected = [
        *fitted["encode_numeric"].cardinalities_,
        *fitted["encode_categorical"].cardinalities_,
    ]
    assert model.cardinalities_ == expected


def test_eager_and_lazy_prediction_agree(fitted, valid):
    np.testing.assert_array_equal(fitted.predict_proba(valid), fitted.predict_proba(valid.lazy()))


def test_extra_passthrough_columns_do_not_matter(fitted, valid):
    extra = valid.with_columns(unused=pl.lit(1))
    np.testing.assert_array_equal(fitted.predict_proba(extra), fitted.predict_proba(valid))


def test_preprocessing_is_one_lazy_plan_over_one_scan(fitted):
    source = pl.scan_parquet(SAMPLE / "valid.parquet").drop("Label")
    plan = fitted[:-1].transform(source)
    assert isinstance(plan, pl.LazyFrame)
    assert plan.explain().count("SCAN") == 1
    encoded = plan.collect()
    assert encoded.columns == NUMS + CATS
    assert all(dtype == pl.Int32 for dtype in encoded.schema.values())


def test_missing_and_zero_numerics_both_map_to_oov(fitted):
    # BARS quirk reproduced with na_values=[0]: null is filled with 0, and 0 is not in the vocab
    schema = {**dict.fromkeys(NUMS, pl.Int32), **dict.fromkeys(CATS, pl.String)}
    nulls = pl.DataFrame({column: [None] for column in schema}, schema=schema)
    zeros = nulls.with_columns(pl.lit(0, dtype=pl.Int32).alias(column) for column in NUMS)
    encoded_missing = fitted[:-1].transform(nulls).select(NUMS)
    encoded_zero = fitted[:-1].transform(zeros).select(NUMS)
    assert encoded_missing.equals(encoded_zero)
    assert not encoded_missing.to_numpy().any()
