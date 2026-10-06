"""BARS ``criteo_x4`` / ``avazu_x4`` splits: MD5-verified CSV -> typed parquet, cached on disk."""

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    """One BARS dataset: where its files are, how to verify them and how to type the CSV."""

    directory: str
    label: str
    md5: dict[str, str]  # reference MD5 of each CSV split
    casts: tuple[pl.Expr, ...]  # applied to the all-String CSV read


NUMERIC = [f"I{i}" for i in range(1, 14)]
CATEGORICAL = [f"C{i}" for i in range(1, 27)]
# Avazu fields after dropping ``id`` and adding the calendar features (see AvazuTimeFeatures)
AVAZU_RAW = [
    "C1", "banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain",
    "app_category", "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]  # fmt: skip
AVAZU_FIELDS = ["hour", *AVAZU_RAW, "weekday", "weekend"]

# Reference MD5s from https://github.com/reczoo/Datasets
CRITEO_X4 = Dataset(
    directory="Criteo_x4",
    label="Label",
    md5={
        "train": "4a53bb7cbc0e4ee25f9d6a73ed824b1a",
        "valid": "fba5428b22895016e790e2dec623cb56",
        "test": "cfc37da0d75c4d2d8778e76997df2976",
    },
    casts=(pl.col("Label").cast(pl.Int8), pl.col(NUMERIC).cast(pl.Int32)),
)
# Avazu has only categorical fields; they stay strings (FuxiCTR reads them with dtype str).
AVAZU_X4 = Dataset(
    directory="Avazu_x4",
    label="click",
    md5={
        "train": "de3a27264cdabf66adf09df82328ccaa",
        "valid": "33232931d84d6452d3f956e936cab2c9",
        "test": "3ebb774a9ca74d05919b84a3d402986d",
    },
    casts=(pl.col("click").cast(pl.Int8),),
)
DATASETS = {"criteo_x4": CRITEO_X4, "avazu_x4": AVAZU_X4}
CRITEO_X4_MD5 = CRITEO_X4.md5


def md5_of(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_parquet(dataset: Dataset, data_root: Path, split: str) -> Path:
    """Parquet for ``split``, built (after verifying the CSV's MD5) if it does not exist yet."""
    directory = data_root / dataset.directory
    parquet = directory / "parquet" / f"{split}.parquet"
    if parquet.exists():
        return parquet
    csv = directory / f"{split}.csv"
    actual = md5_of(csv)
    if actual != dataset.md5[split]:
        msg = f"{csv} MD5 mismatch: expected {dataset.md5[split]}, got {actual}"
        raise ValueError(msg)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    # Read as String and cast: inference would turn all-digit hex ids into integers.
    temporary = parquet.with_suffix(".tmp")
    pl.scan_csv(csv, infer_schema=False).with_columns(*dataset.casts).sink_parquet(temporary)
    temporary.rename(parquet)
    return parquet
