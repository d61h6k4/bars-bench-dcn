import json
import shutil
from pathlib import Path

import pytest

from bars_dcn.bench.data import CRITEO_X4, DATASETS, dataset_parquet, md5_of
from bars_dcn.bench.runner import build_pipeline, load_config, run_seed

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs" / "criteo_x4_dcnv2.toml"


def test_shipped_config_is_the_bars_reference():
    config = load_config(CONFIG)
    reference = json.loads((ROOT / "tests" / "data" / "bars_dcnv2_criteo_x4_001.json").read_text())
    model = config["model"]
    assert model["parallel_hidden_units"] == reference["model"]["parallel_dnn_hidden_units"]
    assert model["num_cross_layers"] == reference["model"]["num_cross_layers"]
    assert model["embedding_dim"] == reference["model"]["embedding_dim"]
    assert model["batch_size"] == reference["training"]["batch_size"]
    assert model["dropout"] == reference["model"]["net_dropout"]
    assert model["embedding_regularizer"] == reference["training"]["embedding_regularizer"]
    assert model["max_grad_norm"] == reference["training"]["max_gradient_norm"]
    assert model["device"] == "mps"
    assert (
        config["preprocessing"]["min_categr_count"]
        == reference["preprocessing"]["min_categr_count"]
    )


def test_avazu_config_is_the_bars_reference():
    config = load_config(ROOT / "configs" / "avazu_x4_dcnv2.toml")
    reference = json.loads((ROOT / "tests" / "data" / "bars_dcnv2_avazu_x4_001.json").read_text())
    model, ref_model = config["model"], reference["model"]
    assert model["parallel_hidden_units"] == ref_model["parallel_dnn_hidden_units"]
    assert model["num_cross_layers"] == ref_model["num_cross_layers"]
    assert model["embedding_dim"] == ref_model["embedding_dim"]
    assert model["batch_norm"] is ref_model["batch_norm"]
    assert model["dropout"] == ref_model["net_dropout"]
    assert model["batch_size"] == reference["training"]["batch_size"]
    assert model["embedding_regularizer"] == reference["training"]["embedding_regularizer"]
    assert model["device"] == "mps"
    assert (
        config["preprocessing"]["min_categr_count"]
        == reference["preprocessing"]["min_categr_count"]
    )


@pytest.mark.parametrize("name", ["criteo_x4", "avazu_x4"])
def test_run_seed_on_the_sample_writes_metrics(tmp_path, name):
    dataset = DATASETS[name]
    parquet = tmp_path / "data" / dataset.directory / "parquet"
    parquet.mkdir(parents=True)
    for split in dataset.md5:
        shutil.copy(ROOT / "tests" / "data" / f"{name}_sample" / f"{split}.parquet", parquet)
    config = load_config(ROOT / "configs" / f"{name}_dcnv2.toml")
    config["model"] |= {
        "parallel_hidden_units": [16],
        "embedding_dim": 4,
        "batch_size": 1024,
        "max_epochs": 3,
        "device": "cpu",
    }
    config["preprocessing"]["min_categr_count"] = 3

    metrics = run_seed(config, seed=7, out_dir=tmp_path / "runs", data_root=tmp_path / "data")

    on_disk = json.loads((tmp_path / "runs" / "seed_7" / "metrics.json").read_text())
    assert on_disk == json.loads(json.dumps(metrics))
    assert on_disk["seed"] == 7
    assert (tmp_path / "runs" / "seed_7" / "tensorboard" / "metrics").is_dir()
    assert 0.55 < on_disk["test"]["auc"] < 1
    assert 0 < on_disk["test"]["logloss"] < 1
    assert 1 <= on_disk["best_epoch"] <= on_disk["epochs_run"] <= 3
    assert on_disk["valid"]["auc"] == on_disk["history"][on_disk["best_epoch"] - 1]["val_auc"]


def test_unknown_dataset_is_rejected():
    config = load_config(CONFIG)
    config["dataset"]["name"] = "movielens"
    with pytest.raises(ValueError, match="unknown dataset"):
        build_pipeline(config, seed=0)


def test_existing_parquet_is_used_without_touching_the_csv(tmp_path):
    parquet = tmp_path / "Criteo_x4" / "parquet"
    parquet.mkdir(parents=True)
    (parquet / "train.parquet").write_bytes(b"x")
    assert dataset_parquet(CRITEO_X4, tmp_path, "train") == parquet / "train.parquet"


def test_csv_with_wrong_md5_is_rejected(tmp_path):
    (tmp_path / "Criteo_x4").mkdir()
    (tmp_path / "Criteo_x4" / "valid.csv").write_text("Label\n0\n")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        dataset_parquet(CRITEO_X4, tmp_path, "valid")
    assert not (tmp_path / "Criteo_x4" / "parquet" / "valid.parquet").exists()


def test_md5_of(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    assert md5_of(tmp_path / "a") == "900150983cd24fb0d6963f7d28e17f72"
