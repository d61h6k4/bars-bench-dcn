from pathlib import Path

import numpy as np
import onnx
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from bars_dcn.bench.latency import (
    benchmark,
    dense_traffic,
    evaluate,
    format_report,
    machine_peaks,
    make_feeds,
    make_session,
    roofline_us,
)
from bars_dcn.estimator import DCNClassifier
from bars_dcn.onnx import to_onnx
from bars_dcn.onnx.quantize import quantize_dense, quantize_embedding
from bars_dcn.preprocessing import MissingFiller, OrdinalEncoder

SAMPLE = Path(__file__).parents[1] / "data" / "criteo_x4_sample"
NUMS = [f"I{i}" for i in range(1, 14)]
CATS = [f"C{i}" for i in range(1, 27)]
HIDDEN = 16


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[Path, pl.DataFrame]:
    train = pl.read_parquet(SAMPLE / "train.parquet")
    pipeline = Pipeline(
        [
            ("fill", MissingFiller(columns=NUMS, fill_value=0)),
            ("encode", OrdinalEncoder(columns=CATS, min_count=3)),
            (
                "model",
                DCNClassifier(
                    cat_columns=CATS,
                    num_columns=NUMS,
                    scalarlens=True,
                    embedding_dim=4,
                    num_cross_layers=2,
                    parallel_hidden_units=[HIDDEN],
                    max_epochs=1,
                    batch_size=256,
                    device="cpu",
                    random_state=0,
                ),
            ),
        ]
    ).fit(train, train["Label"].to_numpy())
    inputs = {**dict.fromkeys(NUMS, "double"), **dict.fromkeys(CATS, "string")}
    path = tmp_path_factory.mktemp("latency") / "model.onnx"
    path.write_bytes(to_onnx(pipeline, inputs).SerializeToString())
    return path, pl.read_parquet(SAMPLE / "valid.parquet").select([*NUMS, *CATS]).head(64)


def test_feeds_follow_the_serving_contract(exported):
    path, requests = exported
    session = make_session(path.read_bytes(), threads=1)
    requests = requests.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("I1")).alias("I1"),
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("C1")).alias("C1"),
    )
    (feed,) = make_feeds(session, requests, batch=3, count=1)
    assert feed["I1"].shape == (3, 1)
    assert np.isnan(feed["I1"][0, 0])
    assert feed["C1"][0, 0] == ""
    label, probabilities = session.run(None, feed)
    assert np.asarray(label).shape == (3,)
    assert np.asarray(probabilities).shape == (3, 2)


def test_dense_traffic_counts_the_weights_of_the_dense_layers(exported):
    traffic = dense_traffic(onnx.load(exported[0]))
    # at least the parallel MLP (n_inputs x HIDDEN) and the cross layers are in there
    assert traffic["weight_bytes"] > 4 * HIDDEN * 52
    assert traffic["flops_per_row"] == pytest.approx(traffic["weight_bytes"] / 4 * 2)


def test_roofline_is_memory_bound_at_batch_one_and_compute_bound_for_large_batches():
    traffic = {"weight_bytes": 23e6, "flops_per_row": 11.5e6}
    peaks = {"bandwidth": 50e9, "gemm_flops": 100e9}
    assert roofline_us(traffic, peaks, 1)["bound"] == pytest.approx(460)
    assert roofline_us(traffic, peaks, 1024)["bound"] == pytest.approx(11.5e6 * 1024 / 100e9 * 1e6)


def test_machine_peaks_are_positive():
    peaks = machine_peaks(1, bandwidth_mb=32, gemm_size=128, runs=3)
    assert peaks["bandwidth"] > 0
    assert peaks["gemm_flops"] > 0


def test_benchmark_reports_latency_ops_and_bound(exported, monkeypatch):
    path, requests = exported
    monkeypatch.setattr(
        "bars_dcn.bench.latency.machine_peaks",
        lambda threads: machine_peaks(threads, bandwidth_mb=32, gemm_size=128, runs=3),
    )
    (result,) = benchmark(path, requests, threads=[1], batch=2, runs=20, warmup=5)
    assert 0 < result["latency_us"]["p50"] <= result["latency_us"]["p99"]
    assert any(op in result["ops_us"] for op in ("Gemm", "FusedGemm", "MatMul"))
    assert result["roofline_us"]["bound"] > 0
    assert "threads=1 batch=2" in format_report([result])


def test_int8_model_is_smaller_and_close_to_fp32(exported, tmp_path):
    path, requests = exported
    quantized = tmp_path / "int8.onnx"
    names = quantize_dense(path, quantized, min_weights=500)
    assert names
    fp32, int8 = path.read_bytes(), quantized.read_bytes()
    assert len(int8) < len(fp32)
    assert (
        dense_traffic(onnx.load(quantized))["weight_bytes"]
        < 0.7 * dense_traffic(onnx.load(path))["weight_bytes"]
    )
    outputs = []
    for model in (fp32, int8):
        session = make_session(model, threads=1)
        (feed,) = make_feeds(session, requests, batch=len(requests), count=1)
        outputs.append(np.asarray(session.run(None, feed)[1])[:, 1])
    np.testing.assert_allclose(outputs[1], outputs[0], atol=0.05)


def test_evaluate_scores_a_partial_last_batch(exported):
    path, _ = exported
    valid = pl.read_parquet(SAMPLE / "valid.parquet").head(100)
    scores = evaluate(path.read_bytes(), valid, batch=64)
    assert scores["rows"] == 100
    assert 0.3 < scores["auc"] < 1
    assert scores["logloss"] > 0


@pytest.mark.parametrize(("kind", "tolerance"), [("fp16", 1e-3), ("int8", 5e-3)])
def test_quantized_embedding_table_is_smaller_and_close_to_fp32(
    exported, tmp_path, kind, tolerance
):
    path, requests = exported
    target = tmp_path / f"{kind}.onnx"
    tables = quantize_embedding(path, target, kind, min_rows=100)
    assert tables == ["embedding_weight"]
    assert target.stat().st_size < path.stat().st_size
    outputs = []
    for model in (path, target):
        session = make_session(model.read_bytes(), threads=1)
        (feed,) = make_feeds(session, requests, batch=len(requests), count=1)
        outputs.append(np.asarray(session.run(None, feed)[1])[:, 1])
    np.testing.assert_allclose(outputs[1], outputs[0], atol=tolerance)


def test_embedding_quantization_needs_a_table(exported, tmp_path):
    with pytest.raises(ValueError, match="no embedding table"):
        quantize_embedding(exported[0], tmp_path / "x.onnx", "int8", min_rows=10**9)
