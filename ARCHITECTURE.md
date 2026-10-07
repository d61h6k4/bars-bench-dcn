# Architecture

Preprocessing, model and postprocessing are **separate sklearn components** composed with
`sklearn.pipeline.Pipeline` (unlike scikit-rank, where the estimator embeds its own
preprocessor).

```
raw polars DataFrame
  -> preprocessing transformers   (fit on train only; own all fitted state)
  -> DCNv2 estimator              (sklearn API; consumes encoded blocks only)
  -> postprocessing (optional)    (e.g. calibration)
```

- **Transformers (passthrough):** `BaseEstimator` + `TransformerMixin`, implemented with polars
  expressions. Each transformer owns a `columns=` parameter; `transform` is
  `X.with_columns(...)`: it replaces its own columns in place and passes every other column
  through, on a `DataFrame` or a `LazyFrame` (same kind out). Fitted state (vocabularies, ...)
  lives here, never in the model. Port ideas from
  `../scikit-rank/src/scikit_rank/preprocessing.py`, don't import it.
- **Embeddings live inside the torch module:** the module takes integer indices (and any numeric
  block), not pre-embedded tensors. Preprocessing includes an ordinal encoder.
- **Estimator:** `DCNClassifier` receives a numeric block (`float32`) and a categorical block
  (integer indices, 0 = OOV/missing). It infers vocab sizes from `X` at fit time and knows
  nothing about raw columns.
- **Lazy by default:** transformers accept `pl.DataFrame | pl.LazyFrame`. `fit` runs lazy
  aggregations (value counts for the min-count vocabulary, mean/std, ...) with the streaming
  engine; `transform` returns the same kind it was given. sklearn's own input validation and
  `set_output` wrapper don't understand `LazyFrame`, so our transformers skip `check_array`
  and handle the output kind themselves; `Pipeline` is agnostic. The estimator is the only
  materialization point: it collects the encoded blocks in compact dtypes (`float32` /
  `int32`, cast to `int64` per batch; ~5.7 GB for the full criteo_x4 train split).
- **Scan from parquet, not CSV:** a CSV scan re-parses text on every `collect()`. Convert the
  splits to parquet once and point `scan_parquet` at them (see PROGRESS.md log for numbers).
  Fitted state (e.g. a vocabulary) lives on the transformer as plain arrays, which is also
  what the ONNX converter reads; `transform` embeds it in the plan as a literal.
- **Composition needs no composite class.** Steps in a plain sklearn `Pipeline` only register
  `with_columns` on the lazy frame; the optimizer fuses them into one node over one scan, so the
  whole preprocessing is invoked once, at the `collect` inside the estimator. (sklearn's
  `ColumnTransformer` can't be used: it rejects a `LazyFrame`, and with eager polars it
  materializes a numpy array. Combining lazy branches with a horizontal concat is also out: no
  column pruning, `CACHE` nodes, 192 s vs 2 s.) The estimator picks its input columns itself
  (`cat_columns`), so the frame may carry any other passthrough columns.
- **Data types:** the BARS pipeline is `LogSquaredBucketizer` (numerics -> int buckets), an
  `OrdinalEncoder(na_values=[0])` for the bucketed numerics and an `OrdinalEncoder` for the string
  categoricals. `OrdinalEncoder` handles one kind per instance (all strings or all integers);
  null (and `""` for strings) is always missing; `na_values` adds more missing values.
- **Fit:** each step's `fit` runs its own pass; that is cheap on parquet (26 vocabularies in
  ~2 s sequentially). Don't use `pl.collect_all` with the default engine (35 s, 16 GB peak).
- **Execution:** declare everything as one lazy plan and invoke it once (one `collect`). The
  lookup must be precompiled: encoding with `replace_strict(dict)` rebuilds a hash map on every
  call (116 ms per 10k-row batch) and peaked at 17-20 GB for 36.7M x 26 string columns, while a
  `pl.Enum` built once at `fit` costs 0.9 ms per batch and encodes all 36.7M rows in one `select`
  in 1.8 s at 7.0 GB peak (result 3.6 GB). So there is no need to execute in column groups, and
  either "encode once into compact arrays, then batch the arrays" or "invoke the plan on each raw
  batch" is affordable. Plans built from `select`/`with_columns` can still be pruned per output
  column; plans combined with a horizontal concat cannot (avoid it).
- **Batching:** plain `torch.utils.data`, not HF `datasets`. A thin batch-level source over
  the encoded arrays (index a whole batch at once, `batch_size=None`), never per-row
  `__getitem__`. If data ever outgrows RAM, memory-map encoded Arrow/parquet behind an
  `IterableDataset`.
- **Torch module:** the DCNv2 `nn.Module` has no sklearn or polars imports, so it can be
  exported on its own.
  `DCNv2(cat_cardinalities, num_features, ...)` takes local per-field integer indices plus an
  optional float block, uses one embedding table with per-field offsets, and returns logits.
- **Intermediates:** keep `float32` / `int32`; never round-trip through `float64` or pandas
  (Criteo_x4 train is 36.7M rows).

## ONNX serving (hard requirement)

The **full pipeline** converts to a single ONNX graph: raw features in (strings allowed),
probability out.

- Transformers are drawn from a **closed set with a known ONNX equivalent** (impute,
  standardize, `floor(ln(x)^2)` bucketize, ordinal encode with a single OOV index and a
  min-count threshold, PLE/bucketize, ...). Don't add a transformer
  without its converter. Planned later (M7): PLE and multihash encoders.
- Each transformer and the estimator register a custom `skl2onnx` converter; the estimator's
  converter embeds the `torch.onnx.export` graph.
- **Parity test is mandatory:** for every shipped config, `pipeline.predict_proba` must match
  `onnxruntime` output within ~1e-5 on real data.

### ONNX implementation

ONNX conversion is a separate step that reads fitted state and emits nodes; it does not shape how
transformers are declared. It rides on `skl2onnx`: `convert_sklearn(pipeline)` walks the plain
sklearn `Pipeline`, and every `bars_dcn` component registers a parser and a converter.

- **Passthrough steps.** skl2onnx models a frame as one `(batch, 1)` variable per column. The
  custom *parser* of a transformer returns the frame it produces: new variables for the columns
  it changes or adds, the *same* variables for all others (they cost nothing and later steps can
  still select by name). The *converter* emits nodes only for the changed columns.
- `LogSquaredBucketizer`: `double` input (NaN = missing -> `fill_value`), `Where`/`Log`/`Mul`/
  `Floor`/`Cast` in float64; verified exactly equal to polars for every integer up to 4M and
  2M random integers up to 2^31-1.
- `OrdinalEncoder`: one `LabelEncoder` (`ai.onnx.ml`) per column, string keys or int64 keys,
  indices 1..n, default 0 (the OOV).
- `AvazuTimeFeatures`: ONNX has no substring op, so the `YYMMDDHH` string is cast to int64 and
  the calendar is computed with integer arithmetic (Sakamoto's weekday); verified against polars
  for every day 2000-2068. The hour must be well-formed (not missing).
- `DCNClassifier`: selects its fields (`feature_names_in_`, fit order) from the frame, `Concat`s
  them into the `(n, fields)` int64 block, inlines the dynamo-exported torch graph, then adds
  `Sigmoid`, `[1 - p, p]`, `ArgMax` and a `Gather` over `classes_`. It needs the field order, so
  it must be fitted on a polars frame.
- Serving contract: one input per column the pipeline reads (columns it never reads, e.g. `id`,
  are not inputs), shape `(n, 1)`, named after the column: `string` with `""` for missing,
  `double` with NaN for missing. Outputs `label` (int64) and `probabilities` (float32, `(n, 2)`).
  Serving needs only the `.onnx` file, `onnxruntime` and numpy. Both BARS pipelines are tested
  end to end against `predict_proba` (1e-5) on real and edge rows (nulls, zeros, negatives,
  unseen tokens, int32 max).
- A stand-alone step whose output column has its input's name gets that input renamed by
  skl2onnx (`hour1`); this cannot happen in a full pipeline, whose outputs are `label` and
  `probabilities`.
- Latency (BARS-size network, whole criteo pipeline, onnxruntime CPU on the M2 Max): 0.19 ms for
  one row, 0.63 ms for 16, 6.4 ms for 256; the graph has 175 nodes. The serving-latency work (int8,
  fused lookup, roofline bench) is in PROGRESS.md (M10) and README.md.
- Opset 20 for the main domain (the fused string lookup needs `StringConcat`; skl2onnx 1.20 stops at 22, so newer opsets are not available), `ai.onnx.ml` 3.

## sklearn component contract

- **Annotations:** sklearn introspects `__init__` and method signatures (`get_params`,
  `set_params`, `clone`, `repr`, `get_metadata_routing`); on 3.14 that evaluates annotations, so
  annotation-only imports stay real imports in sklearn-facing modules. (scikit-rank avoids this
  with `from __future__ import annotations`; we chose real imports because the future import is
  slated for deprecation after Python 3.13's end of life and plain imports work under any
  introspection.)
- **Tags** are declared and enforced: `OrdinalEncoder` accepts strings, categoricals and NaN and
  outputs `int32` (`preserves_dtype = []`); `DCNClassifier` is binary-only
  (`multi_class = False`) and takes non-negative integers (`positive_only`, no NaN).
- **Input contract of `DCNClassifier`:** integer dtype, non-negative, below the per-field
  cardinality seen in `fit`, and the fitted number of fields; violations raise instead of
  silently reading another field's row of the shared embedding table. Cardinalities are
  currently inferred as `max index + 1` in `fit`.
- **Tests:** a targeted contract suite (`tests/test_sklearn_contract.py`) rather than sklearn's
  `check_estimator`, which cannot generate polars frames or encoded blocks: params round trip,
  `set_params` validation, `clone`, `repr`, metadata routing, `fit` not mutating params and only
  adding `*_` attributes, `fit` idempotence, pickle round trip, tags, `Pipeline` composition. A
  guard test requires every sklearn component in the package to be registered. For
  `DCNClassifier` in M4 (numpy input, real training) we will also run
  `estimator_checks_generator` with documented expected failures, as scikit-rank does.

## Training with a fixed validation split

`Pipeline.fit(..., model__eval_set=...)` passes `eval_set` untransformed, which is wrong for
early stopping on the BARS `valid` split. Use the `fit_pipeline(pipe, train, valid)` helper:
it fits the preprocessing steps on train, transforms valid with `pipe[:-1]`, and passes the
result to the estimator. The result is still a plain sklearn `Pipeline`.

### Training loop (`bars_dcn.training`)

Default is an exact replica of FuxiCTR v2.2.0 `BaseModel.fit` (the BARS recipe); every piece is
an estimator parameter. Per batch: BCE-with-logits plus `0.5 * lambda * ||E||^2` on the
embedding table, backward, gradient-norm clip (10), Adam step. Per epoch: reshuffle, validation
AUC in float64, then `PlateauStopper`: a non-improving epoch (AUC < best + 1e-6) multiplies the
LR by 0.1 (floor 1e-6) and counts towards patience 2; an improvement resets the count but not
the LR. The best epoch's weights are restored at the end. Without `eval_set` it trains exactly
`max_epochs` at a fixed LR. A trailing single-row batch is skipped (BatchNorm cannot train on it).
Batches come from a `torch.utils.data.DataLoader` (`bars_dcn.batches`): a seeded batch sampler
owns the shuffle (`torch.randperm` in the main process, so the row order never depends on
`num_workers`), the dataset gathers a whole batch with one tensor index, and `Accelerator.prepare`
places batches on the device (non-blocking from pinned memory on CUDA). `num_workers` /
`prefetch_factor` are parameters; the default 0 is right for these shapes (see PROGRESS.md).
`device="auto"` uses accelerate (MPS here); the fitted model always ends on CPU.
CPU runs are reproducible for a seed; MPS runs are not bit-reproducible.

## End-to-end test data

Fast iteration runs on a small committed sample of `criteo_x4` (13 numeric + 26 categorical),
not the full splits.

- **Files:** `tests/data/criteo_x4_sample/{train,valid,test}.parquet`, 20k / 5k / 5k rows,
  committed to the repo (~1.8 MB) so tests run on a fresh clone and in CI.
- **Provenance:** `scripts/make_sample.py` regenerates them deterministically (seed 2021) from
  the full BARS `train/valid/test.csv` in `data/Criteo_x4/`. It verifies the full files' MD5s
  against the BARS reference values first and keeps the three splits separate.
- **Raw schema, as in the benchmark:** `Label` (int8), `I1..I13` (nullable int32, can be
  negative), `C1..C26` (hex strings, null where missing). Nothing is encoded or imputed in the
  sample; that is the pipeline's job.
- **What the sample covers:** missing numerics and categoricals, string categoricals, a
  negative numeric (`I2`), and OOV (several columns have 40-50% of valid values unseen in the
  sample's train vocabulary). Tests assert that nulls and unseen categories exist, so these
  paths can't silently go untested.
- **Core e2e test:** `fit_pipeline` on train/valid, `predict_proba` on test, ONNX export, and
  onnxruntime parity with the sklearn pipeline (see ONNX serving above). It also asserts a
  loose AUC floor to catch a broken model; the sample is for correctness, never for quality
  numbers.
- **Not the benchmark:** leaderboard and parity numbers always come from the full splits,
  never from the sample.

## Repo layout

Package `bars_dcn`, `src/` layout, built with hatchling.

```
src/bars_dcn/
  model/            # pure torch: cross layers, DCNv2 module. No sklearn/polars imports.
  preprocessing/    # polars-native sklearn transformers (ln^2 bucketizer, ordinal+min-count, ...)
  estimator.py      # DCNClassifier: sklearn API, Accelerate training loop (MPS)
  pipeline.py       # build_pipeline(config), fit_pipeline(pipe, train, valid)
  onnx/             # skl2onnx parsers/converters + to_onnx; importing registers them
  bench/            # dataset registry (paths, MD5s), runner, metrics/reporting
configs/            # one YAML per (model x dataset)
scripts/            # make_sample.py, other one-off tooling
tests/              # unit/, e2e/ (uses the committed sample), onnx/ (parity)
data/               # gitignored full datasets
```

Dependency direction: `bench` -> `pipeline` -> {`estimator`, `preprocessing`} -> `model`.
`onnx` depends on all of them; nothing depends on `onnx` except the export entry point.

## BARS reference run (one config, not the only one)

Reference for parity: `DCNv2_criteo_x4_001_005_c2376d55` (FuxiCTR v2.2.0), stored with its
sources in `tests/data/bars_dcnv2_criteo_x4_001.json`. Our module and pipeline must be general
(all four structures, low-rank mixture, other widths); this config is what M5 parity targets.

- **Model:** `parallel` structure, 3 `CrossNetV2` layers, 5x1000 MLP with BatchNorm (order
  Linear -> BN -> ReLU -> Dropout 0.1), embedding dim 16, final `Linear(624 + 1000, 1)`.
  Cross layer: `x_{l+1} = x_l + x_0 * (W_l x_l + b_l)` with a full `Linear(D, D)`.
- **Init:** embeddings `normal(std=1e-4)`; every `Linear` Xavier-normal with zero bias.
- **Training:** Adam 1e-3, batch 10000, BCE, embedding L2 regularizer 1e-5
  (`(lambda/2) * ||W||^2` per embedding table, added to the loss), grad-norm clip 10,
  early stopping on valid AUC with patience 2 where each non-improving epoch also
  multiplies the LR by 0.1 (min 1e-6), best checkpoint restored.
- **Result:** valid AUC 0.814037 / LogLoss 0.438087; test AUC **0.814514** / LogLoss
  **0.437631**. 20,382,577 parameters.
- **Preprocessing semantics (verified, see below):** all 39 fields are embedded as categoricals.
  Numerics: null -> 0, then `x > 2 ? floor(ln(x)^2) : int(x)`. Vocabulary per field: tokens
  with count >= 10 (train), NA token excluded (`''` for categoricals, `0` for numerics),
  indices 1..n; `__OOV__` = n + 1. Missing, rare and unseen values all go to the single OOV
  index, which has a learned embedding. Index 0 (`__PAD__`) is never produced.

Verification done against the real data and logs (not just the docs):

- All 39 vocabulary sizes reproduced from the full train split (26/26 categoricals, 13/13
  numerics; `floor(log2)` as stated in the BARS dataset README matches only 1/13, the
  FuxiCTR code's `floor(ln(x)^2)` matches all).
- Parameter count reproduced exactly: embeddings 14,571,952 + cross 1,170,000 + MLP 4,629,000 +
  BatchNorm 10,000 + final layer 1,625 = 20,382,577.

Implications for our design: the OOV embedding must be trainable (no frozen `padding_idx`
row for it); the numeric "discretizer" is a bucketize, not a log2; fit-time vocabularies need
the min-count rule with an excluded NA token.

## BARS Avazu reference run (`DCNv2_avazu_x4_001`)

Reference facts live in `tests/data/bars_dcnv2_avazu_x4_001.json`; source: reczoo/Avazu_x4
(MD5s verified, 32,343,172 / 4,042,897 / 4,042,898 rows).

- **FuxiCTR v1.1.1, not v2.2.0.** In v1.1.1 categorical encoders have `__OOV__` = 0 and no
  `__PAD__`, and the embedding has no `padding_idx`, so the OOV row is learned and a field has
  n+1 rows: exactly our `OrdinalEncoder` cardinality. Our parameter count therefore equals BARS's
  directly (73,385,345), with no PAD correction (the Criteo v2.2.0 run needs +1 row per field).
- **Preprocessing:** drop `id`; `hour` (YYMMDDHH) becomes hour-of-day, plus `weekday` (strftime
  `%w`, Sunday = 0) and `weekend`; `min_categr_count = 2`; 24 categorical fields, no numerics.
  Implemented by `AvazuTimeFeatures` (strings out) followed by a string `OrdinalEncoder`; the
  estimator selects the 24 fields with `cat_columns` because the frame still carries `id`.
- **Model:** parallel, 4 cross layers, 4 x 2000 MLP, no BatchNorm, no dropout, embedding
  regularizer 1e-9, embedding dim 16. Same training recipe as Criteo.
- **Behaviour to expect:** BARS overfits fast: valid AUC 0.792978 after epoch 1, then 0.788575
  and 0.776121; early stopping at epoch 3, best epoch 1.
- ONNX for `AvazuTimeFeatures` is not implemented (M5b).

