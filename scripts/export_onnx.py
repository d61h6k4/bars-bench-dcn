"""Fit a config's pipeline and export it to ONNX, for the latency bench.

Writes ``<out>/model.onnx`` and ``<out>/requests.parquet`` (raw test rows to replay). The network
is trained for ``--epochs`` epochs only (default 1): latency does not depend on the weights, but
the vocabulary sizes, and so the table and lookup sizes, are the real ones.

    uv run python scripts/export_onnx.py configs/criteo_x4_dcnv2.toml runs/latency/scalarlens \
        --set 'preprocessing.numeric="scalarlens"' --set 'preprocessing.scalarlens_init="quantile"'
"""

import argparse
from pathlib import Path

import polars as pl

from bars_dcn.bench.data import CATEGORICAL, NUMERIC
from bars_dcn.bench.runner import DATASETS, _split, apply_overrides, build_pipeline, load_config
from bars_dcn.onnx import to_onnx
from bars_dcn.pipeline import fit_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--requests", type=int, default=20_000)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(
        load_config(args.config), [*args.overrides, f"model.max_epochs={args.epochs}"]
    )
    if config["dataset"]["name"] != "criteo_x4":
        msg = "only criteo_x4 is supported (the input column types are hard-coded)"
        raise SystemExit(msg)
    dataset = DATASETS["criteo_x4"]
    train_x, train_y = _split(dataset, args.data_root, "train")
    pipeline = build_pipeline(config, seed=1)
    fit_pipeline(pipeline, train_x, train_y)
    inputs = {**dict.fromkeys(NUMERIC, "double"), **dict.fromkeys(CATEGORICAL, "string")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model.onnx").write_bytes(to_onnx(pipeline, inputs).SerializeToString())
    test = pl.scan_parquet(args.data_root / dataset.directory / "parquet" / "test.parquet")
    test.head(args.requests).collect().write_parquet(args.out / "requests.parquet")


if __name__ == "__main__":
    main()
