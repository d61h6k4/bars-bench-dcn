# bars-bench-dcn

A small, standalone **DCNv2** (Deep & Cross Network v2) with an sklearn API, built to match and then
beat the DCNv2 results on the [BARS-CTR leaderboard](https://openbenchmark.github.io/BARS/CTR/index.html)
(`criteo_x4`, `avazu_x4`), and to serve **fast**: the whole pipeline, from raw columns to a
probability, exports to a single ONNX graph.

- **Model:** DCNv2 (stacked / parallel, full-rank or low-rank mixture cross), plus optional
  numeric embeddings (piecewise-linear bins, [ScalarLens](https://arxiv.org/abs/2609.29182)) and
  multi-hash embeddings.
- **Pipeline:** polars `LazyFrame` end to end; preprocessing, model and postprocessing are separate
  sklearn components in a plain `Pipeline` (see [ARCHITECTURE.md](ARCHITECTURE.md)).
- **Serving:** `bars_dcn.onnx.to_onnx(pipeline, inputs)` gives one ONNX graph (strings in,
  `label` + `probabilities` out), checked against the sklearn pipeline to ~1e-5.
- **Benchmark:** BARS protocol (MD5-verified fixed splits, validation-AUC early stopping, test AUC and
  LogLoss), one TOML config per dataset, `metrics.json` per seed.

Status and the full log of decisions and numbers: [PROGRESS.md](PROGRESS.md).

## Quick start

```bash
uv sync                                                    # Python 3.14
uv run pytest                                              # tests (small committed samples)
uv run ruff check && uv run ruff format --check && uv run ty check
uv run python -m bars_dcn.bench configs/criteo_x4_dcnv2.toml --seeds 2019   # needs data/Criteo_x4/
```

Datasets live under `data/` (gitignored). Overrides use TOML syntax, for example the best
configuration below:

```bash
uv run python -m bars_dcn.bench configs/criteo_x4_dcnv2.toml --seeds 2022 --name scalarlens_q_l2 \
  --set 'preprocessing.numeric="scalarlens"' --set 'preprocessing.scalarlens_init="quantile"' \
  --set 'model.scalarlens_regularizer=1e-4'
```

## Accuracy (criteo_x4, test AUC / LogLoss)

| | AUC | LogLoss |
|---|---|---|
| BARS DCNv2 (FuxiCTR, seed 2019) | 0.814514 | 0.437631 |
| Ours, BARS recipe, 5 seeds (mean) | 0.813633 | 0.438698 |
| Ours, seed 2022 (the ablation seed) | 0.813972 | 0.438288 |
| + fixed LR drop after epoch 4 | 0.814925 | 0.437186 |
| + PLE bins and bucketized embeddings (`both16`) | 0.814994 | 0.437286 |
| + ScalarLens, quantile init, L2 1e-4 (`scalarlens_q_l2`) | **0.815478** | **0.436839** |

`avazu_x4` matches BARS within about 2e-4 AUC. The BARS gap on criteo is explained by when the
learning rate drops (the plateau rule's timing moves AUC by about 0.001). **Caveats:** the rows
below the baseline are single-seed (2022), the L2 value was one a-priori choice and nothing was tuned,
so this is not a leaderboard claim.

## Serving latency (ONNX Runtime CPU, batch 1, preloaded model)

The whole ScalarLens pipeline (26 string lookups, numeric fill, ScalarLens, DCNv2), p50, one thread.
Hetzner `ccx23` (AMD EPYC Milan, AVX2) varies about 25-35% between VMs, so compare only within a
column group.

| Variant | M2 Max | x86 |
|---|---|---|
| fp32 | 487 us | 497 us (first VM) |
| int8 dense layers | 201 us | 210 us (first VM) |
| + MLP `[512]*3` instead of `[1000]*5` | 128 us | 194 us vs 270 us for the int8 baseline (second VM) |
| + MLP `[512]*3`, embedding dim 8 | 113 us | 183 us vs 270 us (second VM) |

What worked: int8 `MatMulInteger` for the large layers (2.3-2.9x, AUC change about 1e-5), one fused
string lookup for all categorical columns (`StringConcat` + a single `LabelEncoder`, -19 us on x86),
BatchNorm folded into the linear layers, a stacked ScalarLens readout, and a vectorized missing-value
fill. What did not: int8 or fp16 embedding tables (smaller file, no latency gain), newer opsets,
the low-rank mixture cross (slower than full-rank at this size: many tiny ops), more than one or two threads.
At batch 1 the dense layers are weight-streaming bound, so bytes read, not FLOPs, set the latency.

Reproduce (needs `data/Criteo_x4/`):

```bash
uv run python scripts/export_onnx.py configs/criteo_x4_dcnv2.toml runs/latency/x \
  --set 'preprocessing.numeric="scalarlens"' --set 'preprocessing.scalarlens_init="quantile"'
uv run python -c "from pathlib import Path; from bars_dcn.onnx.quantize import quantize_dense; \
quantize_dense(Path('runs/latency/x/model.onnx'), Path('runs/latency/x/model_int8.onnx'))"
uv run python -m bars_dcn.bench.latency runs/latency/x/model_int8.onnx runs/latency/x/requests.parquet \
  --threads 1 --eval data/Criteo_x4/parquet/test.parquet
uv run python scripts/speed_sweep.py runs/latency/sweep     # MLP / embedding-size variants, untrained
```

The latency bench also prints the machine's measured bandwidth and GEMM peak and the roofline bound
of the dense layers. On a server without torch, run `python latency.py` directly (it needs only
onnxruntime, onnx, polars, scikit-learn and numpy).

## Not done

- Stage 2 of the speed work: train the narrow-MLP variants (`mlp1000x2`, `mlp512x3`, `mlp256x3`,
  `dim8_mlp512x3`) and report AUC against latency; their latency is measured, their accuracy is not.
- ScalarLens + L2 with the fixed epoch-4 LR drop or PLE bins, and other seeds.
- Categorical lookups and ScalarLens ops still make up about 100 us of the request; the client
  cost of binding 39 inputs is not included in any graph change.

## Layout

```
src/bars_dcn/
  model/          DCNv2, cross layers, MLP, ScalarLens (pure torch)
  preprocessing/  polars passthrough transformers (ordinal, bucketize, PLE, multihash, fill, avazu time)
  onnx/           skl2onnx converters, int8 quantization, graph clean-up
  bench/          runner, datasets, queue, latency bench
  estimator.py    DCNClassifier (sklearn), training.py, batches.py, pipeline.py
configs/          one TOML per dataset      experiments/  queue files of ablations
scripts/          sample builder, ONNX export, speed sweep, RunPod entry point
```
