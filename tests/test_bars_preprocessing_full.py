"""Reproduce BARS preprocessing on the full Criteo_x4 train split (opt-in: -m full_data).

Fits the passthrough pipeline on all 36.7M rows and compares every vocabulary size with the
BARS log, then rebuilds the reference DCNv2 from those sizes and checks its parameter count.
"""

import json
from pathlib import Path

import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from bars_dcn.model import DCNv2
from bars_dcn.preprocessing import LogSquaredBucketizer, OrdinalEncoder

ROOT = Path(__file__).parents[1]
TRAIN_CSV = ROOT / "data" / "Criteo_x4" / "train.csv"
REFERENCE = json.loads((ROOT / "tests" / "data" / "bars_dcnv2_criteo_x4_001.json").read_text())
NUMS = [f"I{i}" for i in range(1, 14)]
CATS = [f"C{i}" for i in range(1, 27)]

pytestmark = [
    pytest.mark.full_data,
    pytest.mark.skipif(not TRAIN_CSV.exists(), reason="full Criteo_x4 CSVs not downloaded"),
]


@pytest.fixture(scope="module")
def preprocessing(tmp_path_factory) -> Pipeline:
    parquet = tmp_path_factory.mktemp("criteo") / "train.parquet"
    # one-off CSV -> parquet so the 39 vocabulary passes are cheap
    pl.scan_csv(TRAIN_CSV, infer_schema=False).with_columns(
        pl.col(NUMS).cast(pl.Int32, strict=False)
    ).sink_parquet(parquet)
    cfg = REFERENCE["preprocessing"]
    pipeline = Pipeline(
        [
            ("bucket", LogSquaredBucketizer(columns=NUMS)),
            (
                "encode_numeric",
                OrdinalEncoder(columns=NUMS, min_count=cfg["min_categr_count"], na_values=[0]),
            ),
            ("encode_categorical", OrdinalEncoder(columns=CATS, min_count=cfg["min_categr_count"])),
        ]
    )
    return pipeline.fit(pl.scan_parquet(parquet).drop("Label"))


def _cardinalities(pipeline: Pipeline) -> dict[str, int]:
    numeric, categorical = pipeline["encode_numeric"], pipeline["encode_categorical"]
    return {
        **dict(zip(numeric.columns_, numeric.cardinalities_, strict=True)),
        **dict(zip(categorical.columns_, categorical.cardinalities_, strict=True)),
    }


def test_all_39_vocabulary_sizes_match_bars(preprocessing):
    # BARS counts an unused __PAD__ row on top of our OOV-inclusive cardinality
    ours = {column: size + 1 for column, size in _cardinalities(preprocessing).items()}
    assert ours == REFERENCE["vocab_sizes"]


def test_reference_model_built_from_our_vocabularies_has_the_bars_parameter_count(preprocessing):
    cfg = REFERENCE["model"]
    cardinalities = [_cardinalities(preprocessing)[column] + 1 for column in NUMS + CATS]
    model = DCNv2(
        cardinalities,
        embedding_dim=cfg["embedding_dim"],
        structure=cfg["model_structure"],
        num_cross_layers=cfg["num_cross_layers"],
        parallel_hidden_units=cfg["parallel_dnn_hidden_units"],
        batch_norm=cfg["batch_norm"],
        dropout=cfg["net_dropout"],
    )
    assert sum(p.numel() for p in model.parameters()) == REFERENCE["expected"]["total_parameters"]
