"""Serving latency of an exported pipeline (onnxruntime CPU) against a roofline.

    uv run python -m bars_dcn.bench.latency model.onnx requests.parquet --threads 1 4

The model is loaded once (preloaded serving): the numbers are steady-state request latency for
``--batch`` rows per call, from raw columns to probabilities. Next to them: the machine's
streaming bandwidth and GEMM peak measured through the same runtime, and the roofline bound of
the dense layers (``max(weight bytes / bandwidth, flops / peak)``), so the gap shows what is
left on the table. Per-op times come from the onnxruntime profiler, which inflates small nodes.
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import polars as pl
from onnx import TensorProto, helper, numpy_helper
from sklearn.metrics import log_loss, roc_auc_score

_DENSE_OPS = ("Gemm", "MatMul", "MatMulInteger")
_FEED_DTYPES = {"tensor(string)": object, "tensor(double)": np.float64, "tensor(float)": np.float32}


def make_session(
    model: str | bytes, threads: int, *, profile: bool = False
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_profiling = profile
    return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])


def make_feeds(
    session: ort.InferenceSession, requests: pl.DataFrame, batch: int, count: int
) -> list[dict[str, np.ndarray]]:
    """``count`` feeds of ``batch`` consecutive rows each (cycling through ``requests``).

    Missing values become ``""`` (strings) or NaN (floats), the serving contract.
    """
    columns = {}
    for spec in session.get_inputs():
        dtype = _FEED_DTYPES[spec.type]
        missing = "" if dtype is object else np.nan
        values = requests[spec.name].to_list()
        columns[spec.name] = np.array([missing if v is None else v for v in values], dtype=dtype)
    n_rows = len(requests)
    feeds = []
    for i in range(count):
        rows = (np.arange(batch) + i * batch) % n_rows
        feeds.append({name: column[rows].reshape(-1, 1) for name, column in columns.items()})
    return feeds


def measure(
    session: ort.InferenceSession, feeds: list[dict[str, np.ndarray]], warmup: int = 200
) -> dict[str, float]:
    """Per-call wall time in microseconds over ``feeds[warmup:]``."""
    for feed in feeds[:warmup]:
        session.run(None, feed)
    times = []
    for feed in feeds[warmup:]:
        start = time.perf_counter_ns()
        session.run(None, feed)
        times.append(time.perf_counter_ns() - start)
    micros = np.asarray(times) / 1e3
    return {
        "p50": float(np.percentile(micros, 50)),
        "p90": float(np.percentile(micros, 90)),
        "p99": float(np.percentile(micros, 99)),
        "mean": float(micros.mean()),
    }


def profile_ops(
    model: str | bytes, threads: int, feeds: list[dict[str, np.ndarray]]
) -> dict[str, float]:
    """Microseconds per call spent in each op type, from the onnxruntime profiler."""
    session = make_session(model, threads, profile=True)
    for feed in feeds:
        session.run(None, feed)
    path = Path(session.end_profiling())
    events = json.loads(path.read_text())
    path.unlink()
    per_op: dict[str, float] = collections.defaultdict(float)
    for event in events:
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time"):
            per_op[event["args"]["op_name"]] += event["dur"] / len(feeds)
    return dict(sorted(per_op.items(), key=lambda item: -item[1]))


def dense_traffic(model: onnx.ModelProto) -> dict[str, float]:
    """Weight bytes read and FLOPs per row by the dense nodes with a constant 2-D weight."""
    weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    total_bytes = flops = 0.0
    for node in model.graph.node:
        if node.op_type not in _DENSE_OPS:
            continue
        for name in node.input:
            weight = weights.get(name)
            if weight is not None and weight.ndim == 2:  # noqa: PLR2004
                total_bytes += weight.nbytes
                flops += 2 * weight.size  # one multiply-add per weight per row
    return {"weight_bytes": total_bytes, "flops_per_row": flops}


def _matmul_graph(rows: int, inner: int, columns: int, repeats: int = 1) -> bytes:
    """``repeats`` chained ``(rows, inner) x (inner, columns)`` products (inner == columns)."""
    weight = np.random.default_rng(0).standard_normal((inner, columns), dtype=np.float32)
    nodes = [
        helper.make_node("MatMul", ["x" if i == 0 else f"h{i}", "w"], [f"h{i + 1}"])
        for i in range(repeats)
    ]
    graph = helper.make_graph(
        nodes,
        "peak",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, inner])],
        [helper.make_tensor_value_info(f"h{repeats}", TensorProto.FLOAT, [rows, columns])],
        [numpy_helper.from_array(weight, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    return model.SerializeToString()


def _best_seconds(session: ort.InferenceSession, feed: dict[str, np.ndarray], runs: int) -> float:
    session.run(None, feed)
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feed)
        best = min(best, time.perf_counter() - start)
    return best


def machine_peaks(
    threads: int, *, bandwidth_mb: int = 512, gemm_size: int = 512, runs: int = 20
) -> dict[str, float]:
    """Streaming bandwidth (B/s) and float32 GEMM peak (FLOP/s) through onnxruntime.

    Bandwidth: a ``(1, K) x (K, N)`` product whose weight (``bandwidth_mb`` MB, beyond any cache)
    is read once per call, exactly the batch-1 access pattern. Peak: a cache-resident square GEMM.
    """
    columns = 4096
    inner = bandwidth_mb * 2**20 // (4 * columns)
    stream = make_session(_matmul_graph(1, inner, columns), threads)
    seconds = _best_seconds(stream, {"x": np.ones((1, inner), np.float32)}, runs)
    bandwidth = inner * columns * 4 / seconds
    repeats = 8
    gemm = make_session(_matmul_graph(gemm_size, gemm_size, gemm_size, repeats), threads)
    seconds = _best_seconds(gemm, {"x": np.ones((gemm_size, gemm_size), np.float32)}, runs)
    peak = repeats * 2 * gemm_size**3 / seconds
    return {"bandwidth": bandwidth, "gemm_flops": peak}


def roofline_us(traffic: dict[str, float], peaks: dict[str, float], batch: int) -> dict[str, float]:
    """Lower bound of the dense layers for one call of ``batch`` rows, in microseconds."""
    memory = traffic["weight_bytes"] / peaks["bandwidth"] * 1e6
    compute = traffic["flops_per_row"] * batch / peaks["gemm_flops"] * 1e6
    return {"memory": memory, "compute": compute, "bound": max(memory, compute)}


def evaluate(
    model: str | bytes, requests: pl.DataFrame, label: str = "Label", batch: int = 1024
) -> dict[str, float]:
    """Test AUC and LogLoss of the exported model on ``requests`` (which carries ``label``)."""
    session = make_session(model, threads=4)
    probabilities = []
    for start in range(0, len(requests), batch):
        chunk = requests.slice(start, batch)
        (feed,) = make_feeds(session, chunk, batch=len(chunk), count=1)
        probabilities.append(np.asarray(session.run(None, feed)[1])[:, 1])
    positive = np.concatenate(probabilities).astype(np.float64)
    target = requests[label].to_numpy()
    return {
        "rows": len(requests),
        "auc": float(roc_auc_score(target, positive)),
        "logloss": float(log_loss(target, positive, labels=[0, 1])),
    }


def benchmark(
    model_path: Path,
    requests: pl.DataFrame,
    *,
    threads: list[int],
    batch: int,
    runs: int,
    warmup: int = 200,
) -> list[dict]:
    """One result per thread count: latency, per-op times, peaks and the roofline bound."""
    model_bytes = model_path.read_bytes()
    traffic = dense_traffic(onnx.load_from_string(model_bytes))
    results = []
    for n in threads:
        session = make_session(model_bytes, n)
        feeds = make_feeds(session, requests, batch, warmup + runs)
        latency = measure(session, feeds, warmup)
        ops = profile_ops(model_bytes, n, feeds[: max(runs // 4, 1)])
        peaks = machine_peaks(n)
        bound = roofline_us(traffic, peaks, batch)
        results.append(
            {
                "threads": n,
                "batch": batch,
                "latency_us": latency,
                "ops_us": ops,
                "peaks": peaks,
                "dense": traffic,
                "roofline_us": bound,
                "dense_bound_vs_p50": bound["bound"] / latency["p50"],
            }
        )
    return results


def format_report(results: list[dict]) -> str:
    lines = []
    for r in results:
        lat, bound, peaks = r["latency_us"], r["roofline_us"], r["peaks"]
        lines += [
            (
                f"threads={r['threads']} batch={r['batch']}: p50 {lat['p50']:.0f} us, "
                f"p90 {lat['p90']:.0f}, p99 {lat['p99']:.0f}"
            ),
            (
                f"  machine: {peaks['bandwidth'] / 1e9:.1f} GB/s stream, "
                f"{peaks['gemm_flops'] / 1e9:.0f} GFLOP/s GEMM"
            ),
            (
                f"  dense layers: {r['dense']['weight_bytes'] / 1e6:.1f} MB weights, "
                f"{r['dense']['flops_per_row'] / 1e6:.1f} MFLOP/row; "
                f"bound {bound['bound']:.0f} us "
                f"(memory {bound['memory']:.0f}, compute {bound['compute']:.0f}) "
                f"= {100 * r['dense_bound_vs_p50']:.0f}% of p50"
            ),
        ]
        total = sum(r["ops_us"].values())
        lines.append(f"  profiled ops (sum {total:.0f} us, inflated by the profiler):")
        lines += [
            f"    {op:24s} {us:7.1f} us {100 * us / total:5.1f}%"
            for op, us in list(r["ops_us"].items())[:8]
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("requests", type=Path, help="parquet with the raw columns the model reads")
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument(
        "--eval", type=Path, default=None, help="parquet with raw columns and Label: report AUC"
    )
    parser.add_argument("--eval-rows", type=int, default=200_000)
    parser.add_argument("--out", type=Path, default=None, help="also write the results as JSON")
    args = parser.parse_args()
    results = benchmark(
        args.model,
        pl.read_parquet(args.requests),
        threads=args.threads,
        batch=args.batch,
        runs=args.runs,
    )
    print(format_report(results))  # noqa: T201
    if args.eval:
        scores = evaluate(args.model.read_bytes(), pl.read_parquet(args.eval).head(args.eval_rows))
        print(  # noqa: T201
            f"eval on {scores['rows']} rows: "
            f"AUC {scores['auc']:.6f}, LogLoss {scores['logloss']:.6f}"
        )
        results[0]["eval"] = scores
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
