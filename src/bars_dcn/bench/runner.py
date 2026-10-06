"""Run one (config, seed): fit the pipeline on train, early-stop on valid, score on test."""

import copy
import json
import logging
import time
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from bars_dcn.bench.data import (
    AVAZU_FIELDS,
    CATEGORICAL,
    DATASETS,
    NUMERIC,
    Dataset,
    dataset_parquet,
)
from bars_dcn.estimator import DCNClassifier
from bars_dcn.pipeline import fit_pipeline
from bars_dcn.preprocessing import (
    AvazuTimeFeatures,
    LogSquaredBucketizer,
    MissingFiller,
    MultiHashEncoder,
    OrdinalEncoder,
    PiecewiseLinearEncoder,
    hash_name,
    ple_columns,
)

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Return a copy of ``config`` with ``section.key=value`` overrides (values are TOML)."""
    result = copy.deepcopy(config)
    for override in overrides:
        key, separator, raw = override.partition("=")
        section, dot, name = key.partition(".")
        if not (separator and dot and name):
            msg = f"override {override!r} must look like section.key=value"
            raise ValueError(msg)
        if section not in result:
            msg = f"override {override!r}: unknown section {section!r}; sections: {sorted(result)}"
            raise ValueError(msg)
        result[section][name] = tomllib.loads(f"value = {raw}")["value"]
    return result


def prepare(config: dict, data_root: Path = Path("data")) -> None:
    """Build (and MD5-verify) the parquet files of the configured dataset."""
    dataset = DATASETS[config["dataset"]["name"]]
    for split in dataset.md5:
        dataset_parquet(dataset, data_root, split)


def build_pipeline(config: dict, seed: int, log_dir: Path | None = None) -> Pipeline:
    """BARS preprocessing for the configured dataset followed by the configured DCNv2."""
    settings = config["preprocessing"]
    min_count = settings["min_categr_count"]
    model_params = dict(config["model"])
    name = config["dataset"]["name"]
    steps: list = []
    if name == "criteo_x4":
        steps, fields = _criteo_steps(settings, model_params, min_count)
    elif name == "avazu_x4":
        steps = [
            ("time", AvazuTimeFeatures()),
            ("encode", OrdinalEncoder(columns=AVAZU_FIELDS, min_count=min_count)),
        ]
        fields = list(AVAZU_FIELDS)
        model_params["cat_columns"] = AVAZU_FIELDS  # the frame still carries ``id``
    else:
        msg = f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}"
        raise ValueError(msg)
    if settings.get("categorical", "ordinal") == "multihash":
        n_hashes = settings["multihash_n_hashes"]
        steps.append(
            (
                "hash",
                MultiHashEncoder(
                    columns=fields, n_hashes=n_hashes, cardinality=settings["multihash_cardinality"]
                ),
            )
        )
        model_params["cat_columns"] = [hash_name(c, k) for c in fields for k in range(n_hashes)]
        model_params["shared_embedding"] = True
    model = DCNClassifier(
        **model_params, random_state=seed, log_dir=str(log_dir) if log_dir else None
    )
    return Pipeline([*steps, ("model", model)])


def _criteo_steps(settings: dict, model_params: dict, min_count: int) -> tuple[list, list[str]]:
    """Criteo preprocessing steps and the encoded index fields they produce.

    ``numeric``: ``bucket`` (BARS: log-squared buckets as embedded ids), ``ple`` (float
    piecewise-linear bins as the numeric block), ``both``, or ``scalarlens`` (raw numerics, missing
    filled with 0, embedded by a ScalarLens inside the model).
    """
    numeric = settings.get("numeric", "bucket")
    if numeric not in ("bucket", "ple", "both", "scalarlens"):
        msg = f"preprocessing.numeric must be bucket, ple, both or scalarlens, got {numeric!r}"
        raise ValueError(msg)
    steps: list = []
    fields = list(CATEGORICAL)
    if numeric in ("ple", "both"):  # before the bucketizer, which replaces the raw numerics
        steps.append(("ple", PiecewiseLinearEncoder(columns=NUMERIC, n_bins=settings["ple_bins"])))
        model_params["num_columns"] = ple_columns
    if numeric == "scalarlens":
        steps.append(("fill", MissingFiller(columns=NUMERIC, fill_value=0)))
        model_params |= {
            "num_columns": NUMERIC,
            "scalarlens": True,
            "scalarlens_init": settings.get("scalarlens_init", "uniform"),
        }
    if numeric in ("bucket", "both"):
        steps += [
            ("bucket", LogSquaredBucketizer(columns=NUMERIC)),
            ("encode_numeric", OrdinalEncoder(columns=NUMERIC, min_count=min_count, na_values=[0])),
        ]
        fields = [*NUMERIC, *CATEGORICAL]
    steps.append(("encode_categorical", OrdinalEncoder(columns=CATEGORICAL, min_count=min_count)))
    if numeric != "bucket":  # the frame also carries columns that are not index fields
        model_params["cat_columns"] = fields
    return steps, fields


def _split(dataset: Dataset, data_root: Path, name: str) -> tuple[pl.LazyFrame, np.ndarray]:
    lazy = pl.scan_parquet(dataset_parquet(dataset, data_root, name))
    labels = lazy.select(dataset.label).collect().to_series().to_numpy()
    return lazy.drop(dataset.label), labels


def run_seed(config: dict, seed: int, out_dir: Path, data_root: Path = Path("data")) -> dict:
    """Train and evaluate one seed; writes ``out_dir/seed_<seed>/metrics.json`` and returns it."""
    dataset = DATASETS[config["dataset"]["name"]]
    train_x, train_y = _split(dataset, data_root, "train")
    valid_x, valid_y = _split(dataset, data_root, "valid")
    test_x, test_y = _split(dataset, data_root, "test")

    target = out_dir / f"seed_{seed}"
    pipeline = build_pipeline(config, seed, log_dir=target / "tensorboard")
    logger.info("seed %d: fitting vocabulary and preprocessing, then training", seed)
    start = time.perf_counter()
    fit_pipeline(pipeline, train_x, train_y, eval_set=(valid_x, valid_y))
    fit_seconds = time.perf_counter() - start

    model = pipeline["model"]
    best = model.history_[model.best_epoch_ - 1]
    probabilities = pipeline.predict_proba(test_x)[:, 1].astype(np.float64)
    metrics = {
        "seed": seed,
        "device": config["model"].get("device", "auto"),
        "best_epoch": model.best_epoch_,
        "epochs_run": len(model.history_),
        "valid": {"auc": best["val_auc"], "logloss": best["val_logloss"]},
        "test": {
            "auc": float(roc_auc_score(test_y, probabilities)),
            "logloss": float(log_loss(test_y, probabilities, labels=[0, 1])),
        },
        "config": config,
        "n_parameters": sum(p.numel() for p in model.model_.parameters()),
        "fit_seconds": fit_seconds,
        "history": model.history_,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("seed %d: valid %s test %s", seed, metrics["valid"], metrics["test"])
    summary = {k: v for k, v in metrics.items() if k not in ("history", "config")}
    logger.info("RESULT %s %s", out_dir.name, json.dumps(summary))
    return metrics
