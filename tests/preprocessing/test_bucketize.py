import numpy as np
import polars as pl
import pytest

from bars_dcn.preprocessing import LogSquaredBucketizer


def _fuxictr_reference(value: float) -> int:
    """FuxiCTR v2.2.0 ``convert_to_bucket`` (fuxictr/datasets/criteo.py), applied after fill 0."""
    if value > 2:
        return int(np.floor(np.log(value) ** 2))
    return int(value)


@pytest.fixture
def numbers():
    return pl.DataFrame(
        {
            "n": [0, 1, 2, 3, 4, 7, 10, 55, 100, 5775, 257675, 23159456, -1, -3, None],
            "keep": list(range(15)),
        },
        schema={"n": pl.Int32, "keep": pl.Int64},
    )


def test_matches_the_fuxictr_reference_formula(numbers):
    out = LogSquaredBucketizer(columns=["n"]).fit(numbers).transform(numbers)
    expected = [_fuxictr_reference(0 if v is None else v) for v in numbers["n"].to_list()]
    assert out["n"].to_list() == expected


def test_known_values():
    frame = pl.DataFrame({"n": [0, 2, 3, 10, 100, None]})
    out = LogSquaredBucketizer().fit(frame).transform(frame)
    assert out["n"].to_list() == [0, 2, 1, 5, 21, 0]
    assert out.schema["n"] == pl.Int32


def test_passthrough_keeps_other_columns_untouched(numbers):
    out = LogSquaredBucketizer(columns=["n"]).fit(numbers).transform(numbers)
    assert out.columns == ["n", "keep"]
    assert out["keep"].to_list() == numbers["keep"].to_list()
    assert out.schema["keep"] == pl.Int64


def test_threshold_and_fill_value_are_configurable():
    frame = pl.DataFrame({"n": [None, 5, 20]})
    out = LogSquaredBucketizer(threshold=10, fill_value=7).fit(frame).transform(frame)
    assert out["n"].to_list() == [7, 5, 8]  # 20 > 10 -> floor(ln(20)^2) = 8


def test_lazy_and_eager_agree_and_lazy_stays_lazy(numbers):
    transformer = LogSquaredBucketizer(columns=["n"]).fit(numbers)
    lazy = transformer.transform(numbers.lazy())
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(transformer.transform(numbers))


def test_non_numeric_columns_are_rejected():
    with pytest.raises(ValueError, match="not numeric"):
        LogSquaredBucketizer().fit(pl.DataFrame({"s": ["a"]}))
