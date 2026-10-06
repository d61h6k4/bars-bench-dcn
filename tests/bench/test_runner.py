import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bars_dcn.bench.data import CRITEO_X4, DATASETS, dataset_parquet, md5_of
from bars_dcn.bench.runner import apply_overrides, build_pipeline, load_config, run_seed

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


@pytest.mark.parametrize(
    "preprocessing",
    [
        {"numeric": "ple", "ple_bins": 4},
        {"numeric": "both", "ple_bins": 4},
        {"numeric": "scalarlens"},
        {"numeric": "scalarlens", "scalarlens_init": "quantile", "categorical": "multihash",
         "multihash_n_hashes": 2, "multihash_cardinality": 500},
        {"categorical": "multihash", "multihash_n_hashes": 2, "multihash_cardinality": 500},
        {"numeric": "ple", "ple_bins": 4, "categorical": "multihash", "multihash_n_hashes": 2,
         "multihash_cardinality": 500},
    ],
    ids=["ple", "both", "scalarlens", "scalarlens+multihash", "multihash", "ple+multihash"],
)  # fmt: skip
def test_criteo_numeric_and_categorical_modes_train(tmp_path, preprocessing):
    parquet = tmp_path / "data" / "Criteo_x4" / "parquet"
    parquet.mkdir(parents=True)
    for split in CRITEO_X4.md5:
        shutil.copy(ROOT / "tests" / "data" / "criteo_x4_sample" / f"{split}.parquet", parquet)
    config = load_config(CONFIG)
    config["model"] |= {
        "parallel_hidden_units": [8],
        "embedding_dim": 4,
        "batch_size": 1024,
        "max_epochs": 2,
        "device": "cpu",
    }
    config["preprocessing"] |= {"min_categr_count": 3, **preprocessing}

    metrics = run_seed(config, seed=1, out_dir=tmp_path / "runs", data_root=tmp_path / "data")

    assert 0.55 < metrics["test"]["auc"] < 1


def test_unknown_numeric_mode_is_rejected():
    config = load_config(CONFIG)
    config["preprocessing"]["numeric"] = "standardize"
    with pytest.raises(ValueError, match="numeric must be"):
        build_pipeline(config, seed=0)


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


class TestOverrides:
    def test_values_are_toml_and_the_original_is_untouched(self):
        config = load_config(CONFIG)
        changed = apply_overrides(
            config, ["model.lr_drop_epochs=[6]", "model.dropout=0.2", 'model.device="cpu"']
        )
        assert changed["model"]["lr_drop_epochs"] == [6]
        assert changed["model"]["dropout"] == 0.2
        assert changed["model"]["device"] == "cpu"
        assert "lr_drop_epochs" not in config["model"]
        assert config["model"]["device"] == "mps"

    @pytest.mark.parametrize("bad", ["model.dropout", "dropout=0.1", "nosuch.key=1", "model.x=[1,"])
    def test_bad_overrides_are_rejected(self, bad):
        with pytest.raises(ValueError, match=r"override|Invalid|expected"):
            apply_overrides(load_config(CONFIG), [bad])


@pytest.fixture
def small_setup(tmp_path):
    """A tiny criteo config and a data root holding the committed sample."""
    parquet = tmp_path / "data" / "Criteo_x4" / "parquet"
    parquet.mkdir(parents=True)
    for split in CRITEO_X4.md5:
        shutil.copy(ROOT / "tests" / "data" / "criteo_x4_sample" / f"{split}.parquet", parquet)
    config = load_config(CONFIG)
    config["model"] |= {"parallel_hidden_units": [8], "embedding_dim": 4, "batch_size": 2048}
    config["preprocessing"]["min_categr_count"] = 3
    small = tmp_path / "small.toml"
    small.write_text(
        "\n".join(
            f"[{section}]\n" + "\n".join(f"{k} = {json.dumps(v)}" for k, v in values.items())
            for section, values in config.items()
        )
    )
    return small, tmp_path / "data"


def _queue(tmp_path, small, data_root, lines, *extra):
    queue = tmp_path / "queue.txt"
    queue.write_text(lines.format(small=small))
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "bars_dcn.bench.queue", str(queue), "--device", "cpu",
         "--data-root", str(data_root), *extra],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )  # fmt: skip


def test_queue_runs_jobs_in_parallel_and_reports_results(tmp_path, small_setup):
    small, data_root = small_setup
    done = _queue(
        tmp_path, small, data_root,
        "# comment\n"
        "{small} --name a --seeds 1 --set model.max_epochs=2 --set model.lr_drop_epochs=[1]\n"
        "{small} --name b --seeds 2 --set model.max_epochs=2\n",
        "--parallel", "2",
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert "QUEUE_START 2 jobs, 2 in parallel" in done.stdout
    assert "QUEUE_DONE" in done.stdout
    results = [line for line in done.stdout.splitlines() if "RESULT" in line]
    assert {line.split("]")[0] for line in results} == {"[a", "[b"}
    first = json.loads((tmp_path / "runs" / "a" / "seed_1" / "metrics.json").read_text())
    assert first["config"]["model"]["lr_drop_epochs"] == [1]
    assert [h["lr"] for h in first["history"]] == pytest.approx([1e-3, 1e-4])
    assert (tmp_path / "runs" / "logs" / "00_a.log").exists()


def test_a_failing_job_makes_the_queue_fail_but_the_others_finish(tmp_path, small_setup):
    small, data_root = small_setup
    done = _queue(
        tmp_path, small, data_root,
        "{small} --name bad --set model.max_epochs=1 --set model.nosuch_parameter=1\n"
        "{small} --name good --seeds 3 --set model.max_epochs=1\n",
        "--parallel", "2",
    )  # fmt: skip
    assert done.returncode != 0
    assert "QUEUE_JOB bad exit=1" in done.stdout
    assert "QUEUE_JOB good exit=0" in done.stdout


def test_a_bad_override_fails_before_any_job_starts(tmp_path, small_setup):
    small, data_root = small_setup
    done = _queue(tmp_path, small, data_root, "{small} --name x --set nosuch.key=1\n")
    assert done.returncode != 0
    assert "QUEUE_START" not in done.stdout
