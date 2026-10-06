import numpy as np
import polars as pl
import pytest

from bars_dcn.preprocessing import PiecewiseLinearEncoder, ple_name


def _frame():
    return pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "other": [9, 9, 9, 9, 9]})


def test_bins_are_a_staircase_with_one_fractional_bin():
    encoder = PiecewiseLinearEncoder(columns=["x"], n_bins=4).fit(_frame())
    assert encoder.edges_[0].tolist() == [0, 1, 2, 3, 4]
    out = encoder.transform(pl.DataFrame({"x": [-5.0, 0.0, 2.5, 4.0, 10.0]}))
    bins = out.select([ple_name("x", k) for k in range(4)]).to_numpy()
    np.testing.assert_allclose(
        bins,
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0.5, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
    )
    assert out[ple_name("x", 0)].dtype == pl.Float32


def test_source_and_other_columns_are_kept():
    out = PiecewiseLinearEncoder(columns=["x"], n_bins=2).fit_transform(_frame())
    assert out.columns[:2] == ["x", "other"]
    assert out["other"].to_list() == [9] * 5


def test_missing_values_are_filled_before_encoding():
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    encoder = PiecewiseLinearEncoder(n_bins=2, fill_value=3).fit(train)
    out = encoder.transform(pl.DataFrame({"x": [None, float("nan"), 3.0]}))
    assert out[ple_name("x", 0)].to_list() == [1.0, 1.0, 1.0]
    assert out[ple_name("x", 1)].to_list() == [0.0, 0.0, 0.0]


def test_duplicate_quantiles_give_fewer_bins_and_constant_columns_none():
    train = pl.DataFrame({"x": [0, 0, 0, 0, 1], "c": [7, 7, 7, 7, 7]})
    encoder = PiecewiseLinearEncoder(n_bins=4).fit(train)
    assert encoder.n_bins_ == [1, 0]
    assert encoder.transform(train).columns == ["x", "c", ple_name("x", 0)]


def test_lazy_matches_eager_and_non_numeric_is_rejected():
    encoder = PiecewiseLinearEncoder(columns=["x"], n_bins=4).fit(_frame().lazy())
    assert encoder.transform(_frame().lazy()).collect().equals(encoder.transform(_frame()))
    with pytest.raises(ValueError, match="not numeric"):
        PiecewiseLinearEncoder(columns=["s"]).fit(pl.DataFrame({"s": ["a"]}))
