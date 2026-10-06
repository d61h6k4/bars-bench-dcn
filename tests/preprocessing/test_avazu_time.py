from datetime import date

import polars as pl
import pytest

from bars_dcn.preprocessing import AvazuTimeFeatures


def _fuxictr(stamp: str) -> tuple[str, str, str]:
    """FuxiCTR v1.1.1 convert_hour / convert_weekday / convert_weekend, verbatim logic."""
    dt = date(int("20" + stamp[0:2]), int(stamp[2:4]), int(stamp[4:6]))
    weekday = int(dt.strftime("%w"))
    weekend = 1 if dt.strftime("%w") in ["6", "0"] else 0
    return str(int(stamp[6:8])), str(weekday), str(weekend)


def test_matches_fuxictr_for_every_day_and_hour_of_the_avazu_period():
    stamps = [f"1410{day:02d}{hour:02d}" for day in range(21, 31) for hour in range(24)]
    frame = pl.DataFrame({"hour": stamps, "id": ["i"] * len(stamps)})
    out = AvazuTimeFeatures().fit(frame).transform(frame)
    expected = [_fuxictr(s) for s in stamps]
    assert list(zip(out["hour"], out["weekday"], out["weekend"], strict=True)) == expected
    assert out["id"].to_list() == frame["id"].to_list()  # passthrough


def test_known_dates():
    # 2014-10-25 is a Saturday, 2014-10-26 a Sunday, 2014-10-27 a Monday
    frame = pl.DataFrame({"hour": ["14102509", "14102600", "14102723"]})
    out = AvazuTimeFeatures().fit(frame).transform(frame)
    assert out["hour"].to_list() == ["9", "0", "23"]
    assert out["weekday"].to_list() == ["6", "0", "1"]
    assert out["weekend"].to_list() == ["1", "1", "0"]


def test_lazy_in_lazy_out_and_same_result():
    frame = pl.DataFrame({"hour": ["14102509", "14102600"]})
    lazy = AvazuTimeFeatures().fit(frame).transform(frame.lazy())
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(AvazuTimeFeatures().fit(frame).transform(frame))


def test_fit_requires_a_string_column():
    with pytest.raises(ValueError, match="must be a String"):
        AvazuTimeFeatures().fit(pl.DataFrame({"hour": [14102509]}))
    with pytest.raises(ValueError, match="must be a String"):
        AvazuTimeFeatures(column="missing").fit(pl.DataFrame({"hour": ["14102509"]}))
