import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from bars_dcn.estimator import DCNClassifier


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.integers(0, [4, 6, 3], size=(50, 3)), rng.integers(0, 2, size=50)


def _model(**kwargs):
    return DCNClassifier(
        embedding_dim=4,
        parallel_hidden_units=[8],
        max_epochs=2,
        batch_size=16,
        device="cpu",
        random_state=0,
        **kwargs,
    )


def test_fit_infers_cardinalities_and_feature_count(data):
    x, y = data
    model = _model().fit(x, y)
    assert model.cardinalities_ == (x.max(axis=0) + 1).tolist()
    assert model.n_features_in_ == 3
    assert model.model_.embedding.num_embeddings == sum(model.cardinalities_)


def test_predict_proba_rows_sum_to_one_and_predict_returns_classes(data):
    x, y = data
    model = _model().fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(model.predict(x)) <= {0, 1}


def test_accepts_numpy_dataframe_and_lazyframe_identically(data):
    x, y = data
    model = _model().fit(x, y)
    frame = pl.DataFrame(x, schema=["a", "b", "c"])
    expected = model.predict_proba(x)
    np.testing.assert_array_equal(model.predict_proba(frame), expected)
    np.testing.assert_array_equal(model.predict_proba(frame.lazy()), expected)


def test_random_state_makes_initialization_reproducible(data):
    x, y = data
    first, second = _model().fit(x, y), _model().fit(x, y)
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))


def test_multiclass_is_rejected(data):
    x, _ = data
    with pytest.raises(ValueError, match="only binary"):
        _model().fit(x, np.arange(50) % 3)


def test_clone_keeps_parameters():
    model = _model(batch_norm=True)
    assert clone(model).get_params() == model.get_params()


def test_cat_columns_selects_and_orders_fields_from_a_polars_frame(data):
    x, y = data
    frame = pl.DataFrame(x, schema=["a", "b", "c"]).with_columns(extra=pl.lit("ignored"))
    model = _model(cat_columns=["c", "a"]).fit(frame, y)
    assert model.n_features_in_ == 2
    assert model.cardinalities_ == [x[:, 2].max() + 1, x[:, 0].max() + 1]
    reference = _model().fit(x[:, [2, 0]], y)
    np.testing.assert_array_equal(model.predict_proba(frame), reference.predict_proba(x[:, [2, 0]]))
    np.testing.assert_array_equal(model.predict_proba(frame.lazy()), model.predict_proba(frame))


def test_cat_columns_requires_a_polars_frame(data):
    x, y = data
    with pytest.raises(ValueError, match="requires a polars"):
        _model(cat_columns=["a"]).fit(x, y)


def test_float_input_is_rejected_as_not_an_index_block(data):
    x, y = data
    model = _model().fit(x, y)
    with pytest.raises(ValueError, match="integer index block"):
        model.predict_proba(x.astype(float))
    with pytest.raises(ValueError, match="integer index block"):
        _model().fit(np.where(x == 0, np.nan, x), y)


def test_negative_indices_are_rejected(data):
    x, y = data
    model = _model().fit(x, y)
    bad = x.copy()
    bad[0, 1] = -1
    with pytest.raises(ValueError, match="non-negative"):
        model.predict_proba(bad)


def test_index_beyond_the_fitted_cardinality_is_rejected_not_read_from_another_field(data):
    x, y = data
    model = _model().fit(x, y)
    bad = x.copy()
    bad[3, 2] = model.cardinalities_[2]  # first index past field 2's range
    with pytest.raises(ValueError, match=r"out of range for field\(s\) \[2\]"):
        model.predict_proba(bad)


def test_wrong_number_of_fields_is_rejected(data):
    x, y = data
    model = _model().fit(x, y)
    with pytest.raises(ValueError, match="expected 3"):
        model.predict_proba(x[:, :2])


def test_predict_before_fit_raises_not_fitted(data):
    with pytest.raises(NotFittedError):
        _model().predict_proba(data[0])


def _frame_with_numeric(data):
    x, y = data
    frame = pl.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2]})
    frame = frame.with_columns(
        n0=pl.Series(np.linspace(0, 1, 50)), n1=pl.Series(np.linspace(1, 0, 50))
    )
    return frame, y


def test_numeric_block_is_trained_and_used_at_predict(data):
    frame, y = _frame_with_numeric(data)
    model = _model(cat_columns=["a", "b", "c"], num_columns=["n0", "n1"]).fit(frame, y)
    assert model.n_numeric_in_ == 2
    assert model.model_.num_features == 2
    shifted = frame.with_columns(pl.col("n0") + 5.0)
    assert not np.allclose(model.predict_proba(frame), model.predict_proba(shifted))


def test_num_columns_requires_explicit_cat_columns(data):
    frame, y = _frame_with_numeric(data)
    with pytest.raises(ValueError, match="explicit cat_columns"):
        _model(num_columns=["n0"]).fit(frame, y)


def test_non_finite_numerics_are_rejected(data):
    frame, y = _frame_with_numeric(data)
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 3).then(None).otherwise(pl.col("n0")).alias("n0")
    )
    with pytest.raises(ValueError, match="finite"):
        _model(cat_columns=["a", "b", "c"], num_columns=["n0"]).fit(frame, y)
