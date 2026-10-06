from pathlib import Path

import polars as pl
import pytest

import bars_dcn

SAMPLE = Path(__file__).parent / "data" / "criteo_x4_sample"
ROWS = {"train": 20_000, "valid": 5_000, "test": 5_000}


def test_package_imports():
    assert bars_dcn.__doc__


@pytest.mark.parametrize(("split", "rows"), ROWS.items())
def test_sample_schema(split, rows):
    frame = pl.read_parquet(SAMPLE / f"{split}.parquet")
    assert frame.height == rows
    assert frame.columns == [
        "Label",
        *[f"I{i}" for i in range(1, 14)],
        *[f"C{i}" for i in range(1, 27)],
    ]
    assert frame.schema["Label"] == pl.Int8
    assert all(frame.schema[f"I{i}"] == pl.Int32 for i in range(1, 14))
    assert all(frame.schema[f"C{i}"] == pl.String for i in range(1, 27))


def test_sample_has_missing_values_and_unseen_categories():
    train = pl.read_parquet(SAMPLE / "train.parquet")
    valid = pl.read_parquet(SAMPLE / "valid.parquet")
    assert train.select("I1", "I12").null_count().sum_horizontal().item() > 0
    assert train.select("C19", "C22").null_count().sum_horizontal().item() > 0
    unseen = valid.filter(pl.col("C3").is_not_null()).join(
        train.select("C3").unique(), on="C3", how="anti"
    )
    assert unseen.height > 0
