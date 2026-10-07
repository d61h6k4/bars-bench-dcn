"""Export untrained variants of the criteo ScalarLens model, for latency (not accuracy).

Latency does not depend on the weights, so each variant is built with random weights; the
preprocessing is fitted on the full train split (real vocabulary sizes), the network on a few rows
(plus one synthetic row that reaches every field's largest index, so the embedding table has its
real size). Per variant it writes ``model.onnx`` (fp32), ``model_int8.onnx`` (int8 dense layers)
and ``model_int8_embint8.onnx`` (also an int8 embedding table) to ``<out>/<variant>/``, and the
replay file ``<out>/requests.parquet``. Time them with ``python -m bars_dcn.bench.latency``.

    uv run python scripts/speed_sweep.py runs/latency/sweep
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from bars_dcn.bench.data import CATEGORICAL, NUMERIC
from bars_dcn.bench.runner import DATASETS, _split, apply_overrides, build_pipeline, load_config
from bars_dcn.onnx import to_onnx
from bars_dcn.onnx.quantize import quantize_dense, quantize_embedding

BASE = {"parallel_hidden_units": [1000] * 5, "embedding_dim": 16}
MIX = {"use_low_rank_mixture": True, "low_rank": 32, "num_experts": 4}
VARIANTS: dict[str, dict] = {
    "base": {},
    "mlp512x3": {"parallel_hidden_units": [512] * 3},
    "mlp256x3": {"parallel_hidden_units": [256] * 3},
    "mlp1000x2": {"parallel_hidden_units": [1000] * 2},
    "dim8": {"embedding_dim": 8},
    "mix": MIX,
    "mlp512x3_mix": {"parallel_hidden_units": [512] * 3, **MIX},
    "dim8_mlp512x3": {"embedding_dim": 8, "parallel_hidden_units": [512] * 3},
    "dim8_mlp512x3_mix": {"embedding_dim": 8, "parallel_hidden_units": [512] * 3, **MIX},
    "dim8_mlp256x3_mix": {"embedding_dim": 8, "parallel_hidden_units": [256] * 3, **MIX},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--rows", type=int, default=50_000, help="rows the network is fitted on")
    parser.add_argument("--requests", type=int, default=20_000)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    config = apply_overrides(
        load_config(Path("configs/criteo_x4_dcnv2.toml")),
        ['preprocessing.numeric="scalarlens"', 'preprocessing.scalarlens_init="quantile"'],
    )
    dataset = DATASETS["criteo_x4"]
    train_x, train_y = _split(dataset, args.data_root, "train")
    pipeline = build_pipeline(config, seed=1)
    *steps, (_, base_model) = pipeline.steps
    encoded = train_x.lazy()
    for _, step in steps:
        encoded = step.fit(encoded, train_y).transform(encoded)

    head = encoded.head(args.rows).collect()
    largest = {
        column: len(categories)  # the highest index: categories are numbered from 1, 0 is OOV
        for column, categories in zip(
            pipeline["encode_categorical"].columns_,
            pipeline["encode_categorical"].categories_,
            strict=True,
        )
    }
    reach = head.head(1).with_columns(
        [pl.lit(i).cast(head.schema[c]).alias(c) for c, i in largest.items()]
    )
    frame = pl.concat([head, reach])
    target = np.append(train_y[: args.rows], 1)

    args.out.mkdir(parents=True, exist_ok=True)
    test = pl.scan_parquet(args.data_root / dataset.directory / "parquet" / "test.parquet")
    test.head(args.requests).collect().write_parquet(args.out / "requests.parquet")
    inputs = {**dict.fromkeys(NUMERIC, "double"), **dict.fromkeys(CATEGORICAL, "string")}
    for name in args.variants:
        model = clone(base_model).set_params(
            max_epochs=1, batch_size=2048, device="cpu", **{**BASE, **VARIANTS[name]}
        )
        model.fit(frame, target)
        directory = args.out / name
        directory.mkdir(exist_ok=True)
        fitted = Pipeline([*steps, ("model", model)])
        (directory / "model.onnx").write_bytes(to_onnx(fitted, inputs).SerializeToString())
        try:
            quantize_dense(directory / "model.onnx", directory / "model_int8.onnx")
            quantize_embedding(
                directory / "model_int8.onnx", directory / "model_int8_embint8.onnx", "int8"
            )
        except ValueError as error:  # no layer is large enough to be worth quantizing
            print(f"{name}: not quantized ({error})", flush=True)
        n_parameters = sum(p.numel() for p in model.model_.parameters())
        print(f"{name}: {n_parameters:,} parameters", flush=True)


if __name__ == "__main__":
    main()
