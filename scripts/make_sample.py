"""Build the small committed criteo_x4 / avazu_x4 samples used by end-to-end tests.

Reads the full BARS splits, verifies their MD5s against the BARS/reczoo reference values, draws
a seeded random row sample from each split (splits are kept separate) and writes parquet. The
data stays raw, exactly as in the source CSVs (criteo: nullable int numerics, hex-string
categoricals, int8 label; avazu: string fields incl. ``id`` and ``hour``, int8 label).

    uv run python scripts/make_sample.py                       # criteo_x4
    uv run python scripts/make_sample.py --dataset avazu_x4
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from bars_dcn.bench.data import DATASETS, Dataset, md5_of

SIZES = {"train": 20_000, "valid": 5_000, "test": 5_000}


def sample_split(path: Path, size: int, seed: int, dataset: Dataset) -> pl.DataFrame:
    # Read everything as String and cast explicitly: schema inference would turn all-digit
    # hex ids (e.g. "32743222") into integers.
    lazy = pl.scan_csv(path, infer_schema=False)
    n_rows = lazy.select(pl.len()).collect().item()
    keep = pl.Series(np.sort(np.random.default_rng(seed).choice(n_rows, size, replace=False)))
    frame = (
        lazy.with_row_index("_row")
        .filter(pl.col("_row").is_in(keep.implode()))
        .drop("_row")
        .collect(engine="streaming")
    )
    return frame.with_columns(*dataset.casts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="criteo_x4")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    data_dir = args.data_root / dataset.directory
    out_dir = Path("tests/data") / f"{args.dataset}_sample"
    for name, expected in dataset.md5.items():
        actual = md5_of(data_dir / f"{name}.csv")
        if actual != expected:
            raise SystemExit(f"{name}.csv MD5 mismatch: expected {expected}, got {actual}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, size in SIZES.items():
        frame = sample_split(data_dir / f"{name}.csv", size, args.seed, dataset)
        out = out_dir / f"{name}.parquet"
        frame.write_parquet(out, compression="zstd")
        print(f"{out}: {frame.height} rows, positive rate {frame[dataset.label].mean():.3f}")


if __name__ == "__main__":
    main()
