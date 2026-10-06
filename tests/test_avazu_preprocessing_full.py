"""Reproduce BARS Avazu preprocessing on the full train split (opt-in: -m full_data).

Fits the Avazu pipeline on all 32.3M rows, rebuilds the reference DCNv2 from the resulting
vocabulary sizes and checks the parameter count against the BARS log (73,385,345).
"""

import json
from pathlib import Path

import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from bars_dcn.bench.data import AVAZU_FIELDS, AVAZU_X4, dataset_parquet
from bars_dcn.model import DCNv2
from bars_dcn.preprocessing import AvazuTimeFeatures, OrdinalEncoder

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data"
REFERENCE = json.loads((ROOT / "tests" / "data" / "bars_dcnv2_avazu_x4_001.json").read_text())

pytestmark = [
    pytest.mark.full_data,
    pytest.mark.skipif(
        not (DATA / "Avazu_x4" / "train.csv").exists(), reason="full Avazu_x4 CSVs not downloaded"
    ),
]


@pytest.fixture(scope="module")
def preprocessing() -> Pipeline:
    train = pl.scan_parquet(dataset_parquet(AVAZU_X4, DATA, "train"))
    min_count = REFERENCE["preprocessing"]["min_categr_count"]
    pipeline = Pipeline(
        [
            ("time", AvazuTimeFeatures()),
            ("encode", OrdinalEncoder(columns=AVAZU_FIELDS, min_count=min_count)),
        ]
    )
    return pipeline.fit(train.drop("click"))


def test_split_sizes_match_bars():
    expected = REFERENCE["expected"]
    for split, key in (("train", "train_rows"), ("valid", "valid_rows"), ("test", "test_rows")):
        rows = pl.scan_parquet(dataset_parquet(AVAZU_X4, DATA, split)).select(pl.len()).collect()
        assert rows.item() == expected[key]


def test_parameter_count_matches_bars(preprocessing):
    cardinalities = preprocessing["encode"].cardinalities_
    assert len(cardinalities) == 24
    model_cfg = REFERENCE["model"]
    model = DCNv2(
        cardinalities,
        embedding_dim=model_cfg["embedding_dim"],
        structure=model_cfg["model_structure"],
        num_cross_layers=model_cfg["num_cross_layers"],
        parallel_hidden_units=model_cfg["parallel_dnn_hidden_units"],
        batch_norm=model_cfg["batch_norm"],
        dropout=model_cfg["net_dropout"],
    )
    # FuxiCTR v1.1.1 has no __PAD__ for categoricals (OOV = 0, a learned row), unlike the v2.2.0
    # Criteo run, so the counts are equal without correction.
    assert sum(p.numel() for p in model.parameters()) == REFERENCE["expected"]["total_parameters"]
