import polars as pl
import pytest
from sklearn.base import clone

from bars_dcn.preprocessing import OOV_INDEX, OrdinalEncoder


@pytest.fixture
def frame():
    return pl.DataFrame(
        {
            "a": ["x", "x", "y", "y", "y", None, "", "z"],
            "b": ["p", "p", "p", "q", "q", "q", "r", "r"],
            "ignored": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


def test_vocabulary_is_ordered_by_count_then_value(frame):
    enc = OrdinalEncoder(columns=["a", "b"], min_count=2).fit(frame)
    assert enc.categories_ == [["y", "x"], ["p", "q", "r"]]
    assert enc.cardinalities_ == [3, 4]


def test_missing_rare_and_unseen_values_all_map_to_oov(frame):
    enc = OrdinalEncoder(columns=["a"], min_count=2).fit(frame)
    new = pl.DataFrame({"a": ["y", "x", "z", None, "", "never-seen"]})
    assert enc.transform(new)["a"].to_list() == [1, 2, 0, 0, 0, 0]
    assert OOV_INDEX == 0


def test_min_count_excludes_rare_values(frame):
    enc = OrdinalEncoder(columns=["a"], min_count=3).fit(frame)
    assert enc.categories_ == [["y"]]


def test_passthrough_replaces_columns_in_place_and_keeps_the_rest(frame):
    out = OrdinalEncoder(columns=["b", "a"], min_count=1).fit(frame).transform(frame)
    assert out.columns == frame.columns  # same order, nothing dropped
    assert out["ignored"].to_list() == frame["ignored"].to_list()
    assert out.schema["a"] == out.schema["b"] == pl.Int32
    assert out.schema["ignored"] == pl.Int64


def test_columns_default_to_all_columns():
    frame = pl.DataFrame({"a": ["x", "x"], "b": ["y", "y"]})
    assert OrdinalEncoder(min_count=1).fit(frame).columns_ == ["a", "b"]


def test_lazy_and_eager_give_identical_results(frame):
    enc = OrdinalEncoder(columns=["a", "b"], min_count=2)
    eager = enc.fit(frame).transform(frame)
    lazy_fit = OrdinalEncoder(columns=["a", "b"], min_count=2).fit(frame.lazy())
    assert lazy_fit.categories_ == enc.categories_
    lazy = lazy_fit.transform(frame.lazy())
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(eager)


def test_float_columns_are_rejected():
    with pytest.raises(ValueError, match="all String or all integer"):
        OrdinalEncoder().fit(pl.DataFrame({"x": [0.5, 1.5]}))


def test_clone_keeps_parameters_and_drops_fitted_state(frame):
    enc = OrdinalEncoder(columns=["a"], min_count=2).fit(frame)
    cloned = clone(enc)
    assert cloned.get_params() == enc.get_params()
    assert not hasattr(cloned, "categories_")


class TestIntegerColumns:
    @pytest.fixture
    def ints(self):
        return pl.DataFrame({"n": [0, 0, 1, 1, 1, 2, -3, None], "other": ["x"] * 8})

    def test_vocabulary_orders_by_count_then_value(self, ints):
        enc = OrdinalEncoder(columns=["n"], min_count=1).fit(ints)
        assert enc.kind_ == "integer"
        assert enc.categories_ == [[1, 0, -3, 2]]

    def test_na_values_are_excluded_and_map_to_oov(self, ints):
        enc = OrdinalEncoder(columns=["n"], min_count=1, na_values=[0]).fit(ints)
        assert enc.categories_ == [[1, -3, 2]]
        assert enc.transform(ints)["n"].to_list() == [0, 0, 1, 1, 1, 3, 2, 0]

    def test_na_values_must_match_the_columns_kind(self, ints):
        with pytest.raises(ValueError, match="na_values must be int"):
            OrdinalEncoder(columns=["n"], na_values=["x"]).fit(ints)
        with pytest.raises(ValueError, match="na_values must be str"):
            OrdinalEncoder(columns=["other"], na_values=[0]).fit(ints)

    def test_mixed_kinds_in_one_encoder_are_rejected(self, ints):
        with pytest.raises(ValueError, match="all String or all integer"):
            OrdinalEncoder(columns=["n", "other"]).fit(ints)

    def test_lazy_and_eager_agree(self, ints):
        enc = OrdinalEncoder(columns=["n"], min_count=1).fit(ints)
        assert enc.transform(ints.lazy()).collect().equals(enc.transform(ints))


def test_empty_string_is_always_missing_for_strings():
    frame = pl.DataFrame({"a": ["", "", "x", "x"]})
    assert OrdinalEncoder(min_count=1).fit(frame).categories_ == [["x"]]


def test_extra_na_values_for_strings():
    frame = pl.DataFrame({"a": ["-1", "-1", "x", "x", "x"]})
    enc = OrdinalEncoder(min_count=1, na_values=["-1"]).fit(frame)
    assert enc.categories_ == [["x"]]
    assert enc.transform(frame)["a"].to_list() == [0, 0, 1, 1, 1]
