# Progress

Status tracker. Update it when a task starts, finishes or changes scope. Newest decisions
and findings go under "Log".

## Done

- [x] CLAUDE.md: Why / What / Success criteria / Environment (Python 3.14, uv, local MPS + Accelerate)
- [x] ARCHITECTURE.md: separate sklearn components, full-pipeline ONNX, `fit_pipeline` helper
- [x] Verified 3.14 binary wheels exist for torch, polars, scikit-learn, skl2onnx, onnxruntime, onnx, accelerate, pyarrow, numpy
- [x] Downloaded Criteo_x1 and Criteo_x4 into `data/` (gitignored); MD5s match BARS references
- [x] M0 scaffold (see Milestones)
- [x] `scripts/make_sample.py` + committed `tests/data/criteo_x4_sample/` (20k/5k/5k, raw schema with nulls and strings, deterministic, MD5-verified source); x1 sample removed

## In progress

(nothing)

## Milestones

Order is risk-first: ONNX is the hardest requirement, so it is spiked before the full
preprocessing set and training are built. Each milestone ends with its verification.

- [x] **M0 Scaffold:** `pyproject.toml` (src layout, hatchling, deps, ruff, pytest), empty
  `bars_dcn` package, `main.py` removed. Verified: `uv run pytest` (5 passed: package import,
  sample schema/rows, nulls and unseen categories in the sample), `ruff check` and
  `ruff format --check` clean.
- [x] **M1 Torch DCNv2:** `bars_dcn.model` with `CrossNetV2`, `CrossNetMix` (low-rank mixture of
  experts), `MLPBlock` and `DCNv2` (all four structures, embeddings inside, optional numeric
  block, returns logits). Verified: cross layers vs naive/reference loops; MLP layer order; final
  width per structure x {full-rank, mixture}; BARS reference config gives exactly **20,382,577**
  parameters; init (embedding std 1e-4, Xavier-normal linears, zero biases); field offsets keep
  fields apart (gradient test); invalid configs rejected; CPU and MPS agree (atol 1e-4, both
  cross types). Mutation-checked: breaking the cross formula, the mixture bias or the field
  offsets each fails the suite. 28 tests pass; ruff clean.
- [x] **M2 ONNX spike:** `OrdinalEncoder` (strings, polars eager/lazy), `DCNClassifier` skeleton
  (sklearn API; `fit` builds the network only, training is M4) and `bars_dcn.onnx.to_onnx`
  (skl2onnx converters for both). Verified: one graph from raw string columns to
  `label` + `probabilities`; onnxruntime matches the sklearn pipeline within 1e-5 (observed
  <= 1.8e-7) on real rows plus all-null, all-empty and never-seen-token rows, for all four
  structures x {full-rank, mixture}, batch sizes 1, 7, 500+; the graph reacts to its inputs;
  null/""/unseen all give the OOV prediction. Mutation-checked (wrong OOV default, reversed
  vocabulary indices, swapped probability columns each fail). At real scale (vocabulary fitted
  on all 36.7M train rows, 909,674 entries, 19.5M-parameter network): parity 1.2e-7 on 5000 real
  rows, 86 MiB model, onnxruntime 0.15 ms (batch 1) / 2.15 ms (100) / 93 ms (5000). The closed
  transformer set holds for this encoder. 57 tests, ruff and ty clean.
- [x] **M3 Transformers (passthrough):** `OrdinalEncoder` (strings or integers, `na_values`,
  Enum-precompiled lookup, passthrough `with_columns`), stateless `LogSquaredBucketizer`
  (`x>2 ? floor(ln(x)^2) : int(x)`, null filled with 0), `DCNClassifier(cat_columns=...)`.
  No composite class (see ARCHITECTURE.md). Verified: bucketizer vs the FuxiCTR reference
  function (negatives, thresholds, null fill, up to 23M); passthrough keeps other columns and
  their dtypes; eager and lazy agree and a lazy pipeline is one plan over one scan; null and 0
  numerics both map to OOV with `na_values=[0]`; full-data (opt-in, `pytest -m full_data`, 20 s):
  fitting the passthrough pipeline lazily on all 36.7M train rows reproduces all 39 BARS
  vocabulary sizes and the reference model built from them has exactly 20,382,577 parameters.
  Mutation-checked (log2 instead of ln^2, no null fill, ignored `na_values`, dropped passthrough,
  ignored `cat_columns` order each fail). New components must be registered in the sklearn
  contract suite (guard enforces it). 108 tests (+2 opt-in); ruff and ty clean.
  Deferred: imputer and standardizer (BARS parity doesn't use them; they matter once float
  numerics enter the model, M7).
- [x] **M4 Estimator + training:** `DCNClassifier`, Accelerate loop on MPS, early stopping,
  `fit_pipeline`. Verify: e2e on the sample (AUC floor, ONNX parity of the trained pipeline).
- [x] **M5 Bench runner, parity:** runner, configs, MD5 checks, per-seed metrics.json done.
  Accepted by the user to proceed with: close to BARS, not matched or beaten. criteo_x4 valid AUC
  0.8130 / 0.8135 / 0.8130 (seeds 2019 / 2020 / 2021) vs BARS 0.814037; best test AUC 0.813938 vs
  0.814514. Training dynamics match BARS epoch by epoch for epochs 1-6 (train loss within 1e-4); the
  gap tracks when the plateau-triggered LR drop happens (BARS dropped after epoch 6, ours after
  7-8). 5 seeds (2019-2020 on MPS, 2021-2023 on RTX 4090), test AUC / LogLoss: 0.813424 / 0.438965,
  0.813938 / 0.438281, 0.813351 / 0.439072, 0.813972 / 0.438288, 0.813479 / 0.438886; mean test AUC
  0.813633 (BARS 0.814514, gap 0.00088), mean LogLoss 0.438698 (BARS 0.437631); mean valid AUC
  0.813231 (BARS 0.814037). Best epoch 8-9 in the CUDA seeds, i.e. the late LR drop again.
- [x] **M5b ONNX for the real pipeline:** both BARS pipelines (criteo: bucketizer -> integer and
  string encoders -> estimator; avazu: time features -> encoder -> estimator with `cat_columns`)
  convert through skl2onnx custom parsers/converters and match `predict_proba` within 1e-5 on real
  and edge rows. 0.19 ms per row at BARS size.
- [x] **M6 avazu_x4:** data MD5-verified, parameter count equals BARS (73,385,345), parity run
  done (seed 2019): test AUC 0.792953 vs BARS 0.793146 (-0.00019), LogLoss 0.371977 vs 0.371865.
- [ ] **M7 PLE + Multihash:** piecewise-linear (PLE) numeric encoding and multihash /
  unified categorical embeddings as preprocessing transformations, each with its ONNX converter,
  evaluated with the same protocol as M5. Design question to settle first: PLE needs fitted bin
  edges, so either PLE is a transformer that emits one feature per bin (model stays generic) or
  the model takes edges as constructor input (couples model and fit-time statistics). Reference:
  scikit-rank reached criteo_x1 AUC 0.8146 with PLE vs 0.8138 `bars_parity`, and avazu_x1 tied at
  0.7646 with multihash but lower seed variance (x1 only; transfer to x4 unknown).
- [ ] **M8 Ablations:** study and ablate DCN variants and tricks, one at a time against the M5
  baseline: low-rank mixture / MoE (the `CrossNetMix` module exists from M1; scikit-rank also has
  top-k MoE), gated cross layers, inner cross layers, other cross variants (e.g. scikit-rank's
  MLDCN layer), and other tricks found while reading. Decided (user): ablations run on a single seed,
  2022 (best validation AUC of the baseline 5 seeds: 0.813577, test 0.813972), and need no ONNX check;
  single-seed differences below ~0.0003 AUC are within seed noise.
- [ ] **M9 Compare with ScalarLens:** arXiv 2609.29182 (numerical embeddings for CTR). Paper read
  (see the 2026-10-06 log entry). Plan: (1) pure-torch `ScalarLens` module + unit tests, (2) wire into
  `DCNv2` as an alternative numeric embedding fed with raw numerics + train ranges, (3) ONNX parity,
  (4) one run on the ablation seed 2022 with the BARS recipe vs the log-squared bucket baseline.
  "Beat the leaderboard" scope stays open.
- [~] **M10 Serving latency (roofline), stopped after stage 1 by the user (2026-10-07):** user goal: the fastest end-to-end ONNX model (raw columns ->
  probability, preloaded session, latency not throughput, x86 target = Hetzner `ccx` dedicated cores via
  the `hcloud` CLI, context `viral-products`), trading accuracy for speed but always measuring both. Best model to
  bench: ScalarLens (`scalarlens_q_l2`). Plan: (1) harness `bars_dcn/bench/latency.py` + `scripts/export_onnx.py`
  [done, local M2 Max], (2) variants: fp16/int8 dense weights, narrower MLP, low-rank cross, ScalarLens
  rank/intervals, embedding quantization, graph fusion of the glue ops; each with latency and AUC (AUC needs a
  training run per variant), (3) repeat on Hetzner x86 (L3 size may change the memory-bound picture).

## Next actions

- [ ] Add an AUC-minus-logloss monitor option (BARS avazu monitors `{AUC: 1, logloss: -1}`)
- [ ] M8: fixed-epoch LR schedule vs plateau trigger (the criteo gap tracks LR-drop timing)
- [ ] Download `avazu_x4` (need link + reference MD5s)

## Open questions

- "Beat the leaderboard": DCNv2 row only, or best model per dataset?
- ScalarLens (M9): paper read; no public code (only an unreferenced "artifact"), so we implement from the paper; the temperature tau and the norm epsilon are not specified (we assume tau=1, eps_n=1e-6).
- Where does `avazu_x4` come from, and what are its reference MD5s?
- Hyperparameter search budget; any GPU beyond local MPS?

## Log

- 2026-10-06: x1 is already preprocessed (scaled numerics, integer category ids, no NaNs), so
  it can't test missing/string paths. Switched the e2e sample to x4, which is raw.
- 2026-10-06: BARS `Criteo_x4_001` preprocessing (numeric bucketing, min_category_count=10
  OOV) is applied at model-prep time, not in the CSV. NOTE: the bucket formula stated in the BARS
  dataset README (floor(log2)) is wrong; see the BARS reference entry below.
- 2026-10-06: Profiled full criteo_x4: 36.67M / 4.58M / 4.58M rows, positive rate 0.256. Missing
  values are real (numerics up to 76% null, categoricals up to 76%), C3 has 8.4M unique values,
  I2 can be negative. x4 sample replaces x1 sample.
- 2026-10-06: Repo layout and milestone order drafted (ARCHITECTURE.md, this file). LazyFrame
  support is part of the design, so out-of-core is no longer out of scope.
- 2026-10-06: Polars LazyFrame experiments (polars 1.44.2, M2 Max, full criteo_x4 train):
  - A LazyFrame is a query plan (source + declared transforms), no data. Declaring transforms
    cost <1 ms; `collect_schema()` reads only the header; the optimizer pushes projection
    down (3 of 40 columns read). Every `collect()` re-executes the whole plan.
  - Fitted state is embedded in the plan as a literal. A 157k-entry vocabulary (C3, min_count
    10) built a plan in 0.01 s; `replace_strict` and a left join each encoded 36.7M rows in
    0.2-0.3 s with 2.4 GB peak RSS on parquet.
  - CSV is the bottleneck: streaming min-count fit over C3 (8.4M uniques) took 5.7 s from CSV
    with ~9.3 GB peak RSS (cumulative peak; the first step is the likeliest source) versus
    0.2 s from parquet. CSV -> parquet (zstd, 1.7 GB) took 7.9 s. Only `train.parquet` has
    been converted so far (ad hoc, in `data/Criteo_x4/parquet/`); valid/test and a committed
    conversion script are still to do.
- 2026-10-06: Composability and batching experiments (full criteo_x4 train, parquet):
  - Separate steps fuse into one `WITH_COLUMNS` node, one scan; selecting one output column
    of a composed plan prunes the rest and matches one-shot results exactly.
  - Fitting 26 vocabularies (909,674 entries total): 1.9 s sequential; `collect_all` default
    engine 35 s / 16 GB; `collect_all` streaming 2.1 s (no gain).
  - Encoding 26 string columns -> int32 (36.7M x 26, 3.6 GB), clean-process peak RSS: one
    select 20.3 GB (also via `sink_parquet`); per column 5.8 GB; 4 row chunks 8.4 GB; joins
    8.5 GB. Times 6-9 s. Decision: execute per column group.
  - One epoch of batch-level shuffled gather (batch 4096) from the in-memory matrix: 2.0 s;
    per-row `Dataset` + `DataLoader`: ~35 s extrapolated from 200k rows (num_workers=0).
  - HF `datasets` 5.1.0 resolves on 3.14 (binary-only) but adds pandas, dill, multiprocess,
    xxhash and an aiohttp stack; no benefit for fixed pre-split data.
  - To verify in M4: `accelerator.prepare` on a `DataLoader(batch_size=None)`; on single-device
    MPS we may only need `prepare(model, optimizer)` and manual batch device moves.
- 2026-10-06: Decisions: we implement everything ourselves and ONNX is first-class, so we write
  our own composite column-transformer step with its own converter (not sklearn's
  `ColumnTransformer`). No `to_parquet` script: the parquet conversion was a one-off learning
  experiment, not part of the project. Full data for benchmarks will be handled when M5 starts.
- 2026-10-06: M0 done. Lint config: ruff `select = ["ALL"]` with a short ignore list (no
  `fix = true`, so `ruff check` never edits files). md5 uses `usedforsecurity=False`.
- 2026-10-06: BARS reference fetched (README log + FuxiCTR v2.2.0 source) and verified:
  test AUC 0.814514 / LogLoss 0.437631, parallel DCNv2, 3 cross layers, 5x1000 MLP, 20,382,577
  params (reproduced exactly). Corrections found: numeric bucketing is floor(ln(x)^2) for x>2
  (not floor(log2)); missing/rare/unseen all map to one learned OOV index; the NA token (0 for
  numerics) is excluded from the vocab. All 39 vocab sizes reproduced from the full train
  split. Reference stored in `tests/data/bars_dcnv2_criteo_x4_001.json`; details in
  ARCHITECTURE.md. Training protocol note: LR x0.1 on every non-improving epoch, stop after 2.
- 2026-10-06: Decisions: embeddings live inside the torch module; preprocessing may contain an
  ordinal encoder.
- 2026-10-06: M1 done. Design choices: one embedding table with per-field offsets (same
  parameter count and L2 penalty as per-field tables, one gather for ONNX); the module returns
  logits (sigmoid lives outside: loss uses BCE-with-logits, ONNX graph adds it at the end);
  `CrossNetMix` shares one gating layer across layers like FuxiCTR; hidden units are required
  (no defaults) for structures that use them; ReLU is the only activation for now. `ruff`
  `max-args` raised to 20 for the flat hyperparameter surface. Not yet checked: ONNX export of
  the module (M2), training behaviour (M4).
- 2026-10-06: Added `ty` (0.0.84, pre-1.0) as a dev dependency; `uv run ty check` is clean. It
  found real problems in `DCNv2` (Optional MLP attributes losing narrowing, untyped buffer), so
  `DCNv2.__init__` now validates and builds each MLP in one branch and `forward` follows which
  MLPs exist (no `structure` branching); behavior unchanged (28 tests, same parameter count).
- 2026-10-06: Plan extended: M7 PLE + Multihash (as preprocessing transformations), M8 ablations
  (MoE / low-rank mixture, gated cross, inner cross, other tricks, learned along the way), M9
  comparison with ScalarLens (arXiv 2609.29182, "ScalarLens: Numerical Embeddings with Stable
  Coordinates and Contextual Responses for CTR Prediction", Yao et al., 2026-09-24). Only the
  abstract has been read: it claims first place in 25 of 27 settings (19 representations, 3
  datasets, 9 backbones, 3 seeds) against DEER, DAES and NaryDis, and argues that externally
  normalized numerics force train/serve statistics to stay in sync (relevant to our ONNX goal).
- 2026-10-06: scikit-rank review: no ONNX anywhere, encoders fitted inside the estimator; useful
  as idea source (PLE, multihash, tuning method, lazy-vs-eager parity test, sklearn compliance
  test, EMA/optimizer toolbox), not as code to import.
- 2026-10-06: M2 done. Decisions and findings:
  - No polars in `bars_dcn.onnx`: serving gets only the `.onnx` file, onnxruntime and numpy.
    Contract: one string tensor `(n, 1)` per column named after it, `""` = missing; outputs
    `label` (int64) and `probabilities` `(n, 2)`; no ZipMap. polars-to-feed conversion lives in
    the tests, as a serving process would build it.
  - torch export: use the supported dynamo exporter (`dynamo=True`, needs `onnxscript`, now a
    dependency) with `dynamic_shapes`; the legacy exporter is deprecated. It folds eval-mode
    BatchNorm into Gemm and exports the mixture layer as plain `Einsum`. Opset 18 (17 forces a
    down-conversion warning). `LabelEncoder` v2 from `ai.onnx.ml` 3.
  - The converter inlines the exported graph into skl2onnx's container (initializers pass as
    `TensorProto`, so weights never become Python lists), then adds Sigmoid -> `[1-p, p]` ->
    ArgMax -> Gather(classes_). Integer class labels only.
  - Untrained weights (embedding std 1e-4) give ~0.5 everywhere, so parity tests randomize
    weights/BN statistics and assert the predictions have spread; otherwise a wrong graph could
    pass.
  - ruff's `TC001/TC003` (move imports under TYPE_CHECKING) are wrong for sklearn estimator
    classes: `BaseEstimator.get_params` (hence `set_params`, `clone`, `repr`) introspects
    `__init__` via `inspect.signature`, and `get_metadata_routing` does the same for `fit` etc.;
    on 3.14 that evaluates annotations at runtime (`NameError`). Tags are not involved. Disabled per file for
    `estimator.py` and `preprocessing/`, with a comment in `pyproject.toml`.
  - (Superseded, see the Enum entry below.) `OrdinalEncoder.transform` originally rebuilt a
    910k-entry dict mapping on every call.
  - Observed once (first e2e run, not reproduced in 8 later runs, exit codes 0): `libc++abi:
    terminating ... recursive_mutex lock failed` at interpreter shutdown, likely thread-pool
    teardown between onnxruntime / torch / polars. Unresolved; watch for it in CI.
  - Known limits: `to_onnx` takes string columns only (M3 extends it); `DCNClassifier.fit` does
    not train (M4); classifier is binary; export of a 19.5M-parameter model takes ~25 s.
- 2026-10-06: sklearn contract settled (decisions: real imports, "approach A"; targeted contract
  tests; honor tags). Investigation: tags are unrelated to the `NameError`; `get_params` (via
  `inspect.signature` on `__init__`) and `get_metadata_routing` (on `fit` etc.) are the failing
  entry points; scikit-rank avoids it with `from __future__ import annotations` and runs the full
  `estimator_checks_generator`. Implemented: tags on `OrdinalEncoder` and `DCNClassifier`;
  `DCNClassifier` input validation (integer dtype, non-negative, per-field range, field count,
  fitted check), which also closes the M1 risk of an out-of-range index reading another field's
  embedding row; `tests/test_sklearn_contract.py` (parametrized contract + registration guard).
  Mutation-checked: an unregistered component, an annotation import moved under TYPE_CHECKING and
  a `fit` that mutates a parameter each fail. 78 tests; ruff and ty clean. Note: estimator
  cardinalities are inferred as max+1, so if the encoder were fitted on different data than the
  estimator, predict now fails loudly (before: silent) -- revisit when `fit_pipeline` exists.
- 2026-10-06: Composition experiments (M3 design input; full criteo_x4 train, 26 string columns):
  - sklearn `ColumnTransformer` is polars-aware for EAGER frames (selects by column name; default
    output is a numpy array, so everything is materialized); `set_output("polars")` fails unless
    the transformer implements `get_feature_names_out`. It REJECTS a `LazyFrame` at input
    validation (`check_array`: "Expected 2D array, got scalar array").
  - Lazy transformers chain automatically in a plain sklearn `Pipeline` (no composite needed for
    sequential steps; separate `with_columns` steps fuse into one node, one scan).
  - Combining lazy branches with `pl.concat(..., how="horizontal")` is a trap: polars inserts
    `CACHE` nodes for the shared scan, `plan.select(col)` does NOT prune the other branches, and
    per-column execution took 192 s (every collect ran the whole plan). Single collect: hconcat
    19.3 GB / 7.5 s vs one `select` of all expressions 17.6 GB / 8.6 s. Per-column pruning works
    only for `select`/`with_columns` plans (verified earlier).
  - skl2onnx's pipeline parser feeds each step's list of output variables into the next step, so
    a step with several outputs (a custom parser) chains natively; its own ColumnTransformer
    parser shows the routing pattern (slice inputs per transformer, concat, merge).
  - Open design question: passthrough transformers (`transform` = `lf.with_columns(...)`,
    untouched columns pass through, plain `Pipeline` + a final column-selection step, ONNX via
    multi-output custom parsers) vs a composite step (expression merging + one routing converter).
- 2026-10-06: Per-batch invocation cost and the Enum fix (full criteo_x4 train, 26 string columns,
  909,674 vocabulary entries, 10k-row raw batch): `replace_strict(dict)` 116 ms/batch (426 s per
  3668-batch epoch) -> precompiled `pl.Enum` + `to_physical` 0.9 ms/batch (1.0 ms through a lazy
  plan), identical output. Bulk, one select over all 36.7M rows: 8.6 s / 17.6 GB peak -> 1.8 s /
  7.0 GB (per column 1.5 s / 4.6 GB); the earlier 20 GB blow-up came from the dict-based
  `replace_strict`, not the streaming engine. `OrdinalEncoder` now builds `dtypes_` (Enum per
  column) at `fit`; 78 tests still pass. Consequence: no column-group execution needed; the
  "declare once, invoke once" design works as a single `select`.
- 2026-10-06: Scoping decision: ONNX is a later, separate step (convert fitted preprocessing to a
  graph and merge with the model graph); it must not drive how transformers are declared. M2's
  skl2onnx converters remain valid but are not binding.
- 2026-10-06: M3 done (see Milestones). Decisions: passthrough `with_columns` transformers in a
  plain Pipeline (no composite class); integer ordinal via int -> string -> Enum (1.5 ms per 10k
  batch incl. bucketing; 1.6 s / 3.0 GB for 36.7M x 13 columns); `na_values` instead of dtype
  magic (BARS excludes the fill value 0 from the numeric vocabulary; default keeps it);
  one dtype kind per encoder instance. The M2 ONNX converters now raise NotImplementedError for
  the passthrough/integer/`cat_columns` cases instead of emitting a wrong graph; M5b covers them.
  Without the unused PAD row our parameter count is 20,382,577 - 39*16 = 20,381,953.
- 2026-10-06: libc++abi shutdown crash: seen twice in ~25 early runs (a passing e2e run and an
  early-stopped `-x` run), then 0 in ~45 runs including 18 deliberate early-stop failures; exit
  codes were 0 where captured. Not reproducible, so not fixed; if it returns in CI, suspect
  thread-pool teardown between polars, torch and onnxruntime at interpreter exit.
- 2026-10-06: M4 done (132 tests, ruff and ty clean). Replica loop in `bars_dcn.training`,
  `DCNClassifier.fit(eval_set=)`, `fit_pipeline`. `PlateauStopper` replays the BARS log exactly
  (best epoch 7, stop at epoch 9, LR 1e-3 -> 1e-6). Sample run: val AUC ~0.727, best epoch 2 of
  4 (restore-best is exercised). Mutation checks (penalty, clip, restore-best, patience, LR
  reduction, single-row skip, shuffle) each fail a test.
  - Speed, BARS-size model, batch 10000, random data: CPU 0.44 s/step, MPS 0.14-0.17 s/step
    (includes setup). Full train is ~3670 steps/epoch, so ~8-10 min/epoch on MPS, ~27 min on CPU.
  - MPS is not deterministic: two same-seed runs gave train losses 0.72884 vs 0.72893 after
    20 steps; CPU is reproducible. Expect run-to-run AUC noise on the full run; not yet
    quantified at full scale.
- 2026-10-06: M5 started. `bars_dcn.bench` (MD5-checked CSV -> parquet, TOML config, runner,
  `python -m bars_dcn.bench`), `device="mps"` is strict. Full criteo_x4 seed 2019 on MPS: epoch 1
  valid AUC 0.806441 vs BARS 0.806450 (first parity datapoint; run still going, started before
  TensorBoard existed so it has no event files).
- 2026-10-06: Observability: `DCNClassifier(log_dir=...)` / runner writes `<seed>/tensorboard/`:
  `metrics/` (epoch scalars) and `spp/epoch_NNN/` (signal-propagation plots, hook based, tags
  `Signal_Prop_{Forward,Backward}/Layer_{Std,Dist}`, step = layer index). View with
  `uv run tensorboard --logdir runs/<config>`. Note: the pasted SPP spec was written for
  deformable-conv DCNv2 (offset/mask branches); our DCNv2 is Deep & Cross Network v2 and has
  none, so those two requirements do not apply. SPP probes in eval mode on the first 2048 train rows.
- 2026-10-06: SPP reworked to be hook-only (no model changes): per-branch tags
  (`Signal_Prop_{Forward,Backward}/{embedding,cross,stacked,parallel,head}/Layer_{Std,Dist}`),
  exact cross residual stream x0..xL, post-ReLU deep points with `Zero_Fraction`; unbranched tags
  kept (global order as step). Epoch 0 probe (`spp/epoch_000`, initial state) added; logging
  verified not to change training. Finding: with FuxiCTR init everything is ~1e-4 at init and the
  cross layers are ~identity (adds ~1e-8); embeddings grow to ~0.02 within epoch 1. PyTorch-default
  embedding init (N(0,1)) gives order-1 signal and logit std ~2.5 at init: M8 ablation, not
  assumed better. The seed-2019 run in flight (started before this) has the old flat layout.
- 2026-10-06: M5 criteo_x4 parity, 2 seeds (stopped by the user to move on; protocol said 5).
  Seed 2019: valid 0.813018 / test 0.813424 AUC, test LogLoss 0.438965 (best epoch 9, LR drop at 9,
  77 min on MPS) vs BARS 0.814037 / 0.814514 / 0.437631. About 0.001 AUC below BARS; seed 2020
  dropped the LR a epoch earlier and is higher, so the LR-drop epoch drives the spread. Parity
  within seed noise is NOT established. Epochs 1-5 matched the BARS log within ~1e-4.
- 2026-10-06: M6 started. avazu_x4 downloaded from reczoo (MD5 train de3a2726, valid 33232931,
  test 3ebb774a; all verified). BARS Avazu uses FuxiCTR v1.1.1: OOV=0, no PAD, learned OOV row, so
  our param count equals BARS exactly: 73,385,345 (`pytest -m full_data`). Added
  `AvazuTimeFeatures`, dataset registry (`bars_dcn.bench.data`), `configs/avazu_x4_dcnv2.toml`,
  `scripts/make_sample.py --dataset avazu_x4` and an avazu e2e runner test. RunPod access
  exists (MCP tools); the remaining criteo seeds run there (see below).
- 2026-10-06: Training batches now come from a torch DataLoader (`bars_dcn.batches`) with Accelerate
  placing them on the device; `num_workers`/`prefetch_factor` exposed. Same shuffle as before
  (verified equal to the old randperm sequence, with 0 and 2 workers); results independent of
  `num_workers` (a multi-process loader draws a base seed from the global RNG and shifted the dropout
  masks until the loader got its own generator). Measured, 36.7M x 39 int32: gather 1.8 ms/batch
  with a torch tensor index in-process, vs 16.5 ms with the old numpy fancy index; workers add IPC
  overhead (2-3 ms/batch at 10M rows), so they do not help here and stay at 0 by default. On MPS the
  step is compute bound (~140-260 ms), so the loader never mattered much there. Not yet measured on
  CUDA. Correction: an earlier "GPU 46% => loader bound" reading from the RunPod pod was taken while
  the pod was still unzipping data and meant nothing.
- 2026-10-06: RunPod lessons: pin allowedCudaVersions (torch 2.14 ships CUDA 13 libs; a CUDA 12.8
  host fails with "driver too old"), make the start script fail-fast and non-restarting, and note
  that PyPI download speed varies hugely by host (18m41s vs 40s for the same 75 packages).
- 2026-10-06: M6 avazu_x4 parity, seed 2019 on MPS (BARS used one seed too): valid 0.792867 /
  0.372029, test 0.792953 / 0.371977 vs BARS valid 0.792978 / 0.371967, test 0.793146 / 0.371865.
  Epoch curve tracks BARS (0.7929, 0.7886, 0.7758 vs 0.7930, 0.7886, 0.7761), best epoch 1,
  stopped at epoch 3 like BARS. Parity holds within ~2e-4 AUC.
- 2026-10-06: criteo seed 2021 on RunPod RTX 4090 (CUDA 13.0, 64 vCPU host): 33 steps/s, ~2 min/epoch,
  ~21 min per seed incl. 35 s of data prep; best epoch 9, valid AUC 0.812991. Train loss per epoch
  vs the BARS log: 0.4590/0.4514/0.4498/0.4485/0.4476/0.4468 (BARS) vs 0.4594/0.4516/0.4499/0.4486/
  0.4476/0.4467 (ours). BARS avazu monitors AUC minus logloss; ours monitors AUC only (no effect
  there, best epoch was 1 either way). Do not claim better-than-BARS: the best seed is 0.0005 below.
- 2026-10-06: M5b done. skl2onnx kept (the point of the sklearn API): a custom parser per
  transformer returns the frame (changed columns = new variables, the rest pass through). I first
  started a hand-written walker instead and dropped it when the user objected. skl2onnx friction
  points, all handled: the default parser assumes one matrix per step; the classifier shape
  calculator asserts a single input; initial_types must list exactly the columns read (ORT requires
  every declared input); a stand-alone step renames an input that shares its output's name.
  ONNX has no substring op, so Avazu's hour string is parsed with Cast + integer calendar math.
  Exact-equality checks: bucketizer vs polars over 6M integers incl. 2^31-1; weekday/hour/weekend
  vs polars for every day 2000-2068. Removed `test_unsupported` (those cases are now supported).
  `DCNClassifier` now records `feature_names_in_` when fitted on a polars frame.
- 2026-10-06: M8 infrastructure: `lr_drop_epochs` (fixed LR schedule replacing the plateau trigger),
  `python -m bars_dcn.bench --set section.key=value --name N`, `python -m bars_dcn.bench.queue FILE
  --parallel K` (runs a file of jobs concurrently, echoes only epoch/RESULT lines, full logs in
  runs/logs/), `scripts/pod_run.sh` (pod entrypoint: install, data, queue). Batch 1
  (`experiments/m8_lr_schedule.txt`): LR drop after epoch 4/5/6/7 on criteo, seeds 2021-2023.
- 2026-10-06: criteo seeds 2021-2023 finished on the RunPod 4090 (~21 min fit each, plus ~5 min CPU
  test prediction). The same pod was re-pointed (update-pod + restart) at the M8 batch 1 queue,
  PARALLEL=4.
- 2026-10-06: M7 step 1 (float numeric block: `DCNClassifier(num_columns=...)`, batches/training carry
  a numeric tensor) and step 2 (`PiecewiseLinearEncoder`: quantile edges, de-duplicated, additive
  `{col}_ple{k}` float32 columns, missing filled before encoding) done; ONNX for both is step 3.
- 2026-10-06: M7 step 3 done: ONNX for `PiecewiseLinearEncoder` (float64 Sub/Div/Clip per column, Split
  into `(batch,1)` float32 columns, bin values exactly equal to polars) and for estimators with a numeric
  block (graph inputs `x_cat` int64 + `x_num` float32, both with a dynamic batch). Pipeline parity
  on the criteo sample incl. missing numerics. Next: multihash (step 4).
- 2026-10-06: M7 step 4 done: `MultiHashEncoder` (per column and hash random `(a, b)`, int64
  `(a*i+b) mod M`, additive `{col}_h{k}` columns, ONNX Mul/Add/Mod), `DCNv2(shared_embedding=True)` /
  `DCNClassifier(shared_embedding=True)` (one `max(card)`-row table, no offsets). ONNX parity on a
  hashed pipeline. Next: step 5, bench config (`numeric = bucket|ple|both`, `categorical = ordinal|multihash`).
- 2026-10-06: M7 step 5: bench config `preprocessing.numeric = bucket|ple|both` (+ `ple_bins`) and
  `preprocessing.categorical = ordinal|multihash` (+ `multihash_n_hashes`, `multihash_cardinality`) for
  criteo; `DCNClassifier(num_columns=...)` may be a function of the frame's column names
  (`ple_columns`). First batch: `experiments/m7_ple_multihash.txt` (seed 2021), to run on the pod after
  M8 batch 1. Note multihash with `embedding_dim=8` x 2 hashes keeps 16 dims per field.
- 2026-10-06: Parallel jobs on one RTX 4090 pod (12 vCPU) are slower than sequential: 4 jobs took 24.5 min
  for epoch 1 each (~6 min of pod time per job-epoch) vs ~2 min for a single job (CPU-bound: pod CPU 99%,
  GPU 50%). Restarted M8 batch 1 with PARALLEL=1. Rule: one criteo job at a time per pod unless the
  host has many more vCPUs. Also: variants that share a seed are identical until the first LR drop, so
  `lrdrop4..7` duplicate epochs 1-4 (checkpoint/resume would avoid it).
- 2026-10-06: M8 batch 1 reduced to seed 2022 only (4 jobs: LR drop after epoch 4/5/6/7); M7 batch 1 also on
  seed 2022. Pod restarted on the new queue.
- 2026-10-06: ScalarLens paper read (arXiv 2609.29182, Yao et al.). Method: per numeric field, K=16
  learned monotone intervals over the training range [l,u] (widths = softmax(w/tau) with floor eps=1e-4),
  the value is clipped to [l,u] and linearly interpolated between two learned d=16 vectors of the
  interval ends ("stable coordinate", depends on the scalar only). Context: the N numeric tokens and C
  categorical embeddings are RMS-normalized (no affine); a per-field head gives drive delta_f and gate g_f
  (m=8); a shared low-rank operator U (M x r), V (r x M), M = F*m, r=16, runs T=3 bounded steps
  s <- (1-a_t) s + a_t tanh(delta + g * (sU)V), a_t = sigmoid(learned); the readout of field i is a SiLU
  MLP (width 32) over [normalized coordinate, gate_i, s_i^(1..T)] times a positive learned gain, giving a
  d=16 numeric token. Categorical embeddings go to the unchanged backbone. Everything is standard ops,
  so ONNX export is plausible (interval lookup via comparisons, no searchsorted needed). Inputs are raw
  numerics (no log / z-score), ranges stored with the model, inference clips to them.
  Their evidence uses a different, weaker protocol (batch 4096, MLP 256-128-64, no BN/dropout, seeds
  2026-2028): Criteo DCNv2 test AUC log-squared bucket 0.8112, PLE-A 0.8121, DEER 0.8122, ScalarLens 0.8133
  (+0.0021 over log-squared bucket, +0.0011 over DEER; std <= 0.0002). The BARS recipe is already at
  0.8136-0.8145 with bucketized numerics, so the gain may not transfer; the numeric path is only 13 of
  39 fields. Their split sizes (36.67M / 4.58M / 4.58M) equal criteo_x4.
- 2026-10-06: M9 steps 2-3 done: `DCNv2(scalarlens=True)` / `DCNClassifier(scalarlens=True)` (raw numeric block
  embedded by `ScalarLens` into one token per field; training min/max recorded at fit), `MissingFiller`
  transformer (null/NaN -> constant, float64; ONNX IsNaN/Where), ONNX estimator export accepts double
  numeric columns (cast to float32 like at fit), bench `preprocessing.numeric = "scalarlens"`. ONNX parity
  of a full pipeline with missing and far-out-of-range numerics. Step 4 (the run) is queued in
  `experiments/m9_scalarlens.txt`, after M8 and M7 on the pod.
- 2026-10-06: M8 batch 1, first result (seed 2022, LR x0.1 after epoch 4, `lrdrop4`): valid AUC 0.814606,
  test AUC 0.814925, test LogLoss 0.437186, best epoch 6 of 8 (baseline seed 2022: 0.813577 / 0.813972 /
  0.438288; BARS: 0.814037 / 0.814514 / 0.437631). +0.00095 test AUC vs our baseline, +0.00041 vs BARS.
  Single seed, drop epoch chosen on validation: promising, not yet a leaderboard claim (needs the
  5-seed protocol with the chosen schedule). Epoch time alone on the 4090: ~2 min; fit 17 min + 5 min
  test prediction. lrdrop5-7 running.
- 2026-10-06: M8 batch 1, `lrdrop5` (seed 2022, LR x0.1 after epoch 5): valid 0.814472, test 0.814787, test LogLoss
  0.437420, best epoch 7 of 9. `lrdrop4` 0.814925 vs `lrdrop5` 0.814787: earlier drop slightly better,
  within single-seed noise. `lrdrop6` at epoch 8: valid 0.814024 (drop after epoch 6, best so far 0.814123).
- 2026-10-06: Plan agreed with the user: run M7 and M9 first (`experiments/m7_m9.txt`: ple16, both16, multihash1m,
  scalarlens; seed 2022, plateau LR rule, sequential), then the 5-seed protocol with the best combination
  (including the best fixed LR drop). PLE fit on the full train split: 31 s, 128 bin columns in total
  (low-cardinality fields get fewer bins), i.e. ~19 GB float32 numeric block.
- 2026-10-06: scikit-rank review for M7/M9: its "PLE" is a per-feature learned embedding of the bins (Linear +
  ReLU + feature dropout 0.1, 32 bins; criteo_x1 test AUC 0.8138 -> 0.8146, 5 seeds), not raw bins. Decided
  (user): our `ple` = classic untrained PLE (raw bins as numeric block); the trained version is covered by
  ScalarLens (interpolation between learned knot vectors == PLE + linear, plus learned boundaries).
  Found: on criteo_x4 train, with equal-width initial intervals over the raw range, >99.9% of the rows of 11 of
  13 numeric fields fall in interval 0 (I10: 71%), so the paper-faithful init starts with a near-constant
  coordinate. Added `scalarlens_init = "quantile"` (boundaries start at training quantiles, still learnable)
  next to the paper's "uniform"; the queue `experiments/m7_m9.txt` now runs both (scalarlens_q, scalarlens),
  then multihash1m, ple16, both16. Gated/inner cross and top-k MoE: deprioritized (see M8); DCN-Mix first as a
  signal for the cross-layer family.
- 2026-10-06: M8 batch 1 complete (seed 2022; test AUC / LogLoss, best epoch): lrdrop4 0.814925 / 0.437186 (6),
  lrdrop5 0.814787 / 0.437420 (7), lrdrop6 0.814466 / 0.437697 (7), lrdrop7 0.813972 / 0.438288 (8). lrdrop7 is
  bit-identical to the plateau baseline of seed 2022 (the plateau rule dropped after epoch 7 for that seed), a
  reproducibility check across two pods. Earlier drop = better, monotonically (0.814925 > 0.814787 > 0.814466 >
  0.813972); drop after epoch 6 (BARS's own timing) reaches 0.814466 vs BARS 0.814514. The pod then ran
  idle 18:55-19:02 UTC before being pointed at `experiments/m7_m9.txt` (restart ~19:02).
- 2026-10-06: The M7/M9 queue crashed at start (19:04 UTC): `shlex.split` in the queue strips the double quotes of
  `--set k="str"`, leaving a bare word that is not TOML. Fixed by single-quoting those `--set` values in the queue
  files, and added `tests/bench/test_experiments.py` (every line of every experiments/*.txt must parse, apply and
  build a pipeline). The pod idled ~21 min before the crash was seen (19:04-19:25).
- 2026-10-06: M9 first result, `scalarlens_q` (ScalarLens, quantile-initialized boundaries; seed 2022; plateau LR rule;
  raw numerics, missing -> 0): valid 0.813257, test 0.813533, test LogLoss 0.439017, best epoch 8 of 10, 20,417,425
  parameters (+35k), ~3.3 min/epoch (baseline ~2). Baseline seed 2022: 0.813577 / 0.813972 / 0.438288 -> -0.00044
  test AUC, +0.00073 LogLoss; BARS 0.814514 -> -0.00098. Val AUC was ahead of the baseline in epochs 1-7 (+0.0023 at
  epoch 1, shrinking to +0.0008 at epoch 7) and fell behind after the LR drop; train loss falls much faster than
  baseline (0.419 vs 0.4275 at epoch 10) while val LogLoss rises (0.4436): overfits, nothing regularizes it (L2 only
  on the categorical table). Not yet tested: ScalarLens with the epoch-4 fixed LR drop (it peaks early), or with
  regularization. `scalarlens` (uniform init) epoch 1: val AUC 0.804772 (baseline 0.806365, quantile 0.808703).
- 2026-10-06: M9 `scalarlens` (paper's equal-width initial intervals; seed 2022; plateau LR rule): valid 0.813555, test
  0.813898, test LogLoss 0.438118, best epoch 4 of 6 (the plateau rule dropped the LR already after epoch 2; peak 2
  epochs later, then overfits: val LogLoss 0.4463 at epoch 6). vs baseline 0.813972 / 0.438288: -0.00007 AUC (noise),
  -0.00017 LogLoss; vs `scalarlens_q` 0.813533: +0.00037. Conclusion so far: ScalarLens (either init) does not beat the
  log-squared buckets under the BARS recipe on this seed; results are confounded by the plateau rule's LR-drop timing
  (+-0.001), a fixed-schedule comparison was not run (user declined). `multihash1m` epoch 5 val AUC 0.811281 (baseline
  0.811634).
- 2026-10-06 (night plan, final): no 5-seed run (user). The user asked for one more experiment: ScalarLens with quantile
  boundaries plus regularization. Added `scalarlens_regularizer` (L2 on the ScalarLens parameters except the boundary
  logits, applied like the embedding L2; logged train loss includes the penalty, ~0.04 at 1e-4) and `scalarlens_dropout`
  (whole numeric tokens dropped while training). Queue `experiments/m9_scalarlens_reg.txt`: `scalarlens_q_l2` (1e-4) and
  `scalarlens_q_drop` (0.1), seed 2022, plateau rule, to run on the same pod right after the M7+M9 queue (~22:35 UTC), then
  record the results and DELETE the pod (deadline 02:30 UTC).
- 2026-10-06: M7 results (seed 2022, plateau LR rule; baseline valid 0.813577 / test 0.813972 / LL 0.438288, 20,381,953 params):
  * `multihash1m` (2 hashes into a 1M-row shared table, dim 8 per hash): valid 0.813212, test 0.813709, LL 0.438556, best
    epoch 8 of 10, 13,810,625 params (-32%): -0.00026 test AUC, +0.00027 LogLoss for a third fewer parameters.
  * `ple16` (classic untrained PLE: 16 quantile bins per numeric field as a raw float block, 128 columns, next to the
    categorical embeddings; the 13 numerics are NOT bucketized): valid 0.814364, test 0.814703, LL 0.437709, best epoch 9
    of 11, 20,005,185 params: +0.00073 test AUC, -0.00058 LogLoss vs baseline; vs BARS (0.814514 / 0.437631):
    +0.00019 AUC, +0.00008 LogLoss. The plateau rule dropped the LR after epoch 8 (baseline: after 7), so drop timing
    is not the same, but a later drop did not hurt. Single seed; untested with the epoch-4 fixed drop.
  * `both16` (PLE bins + bucketized embeddings): running; epoch 6 (first epoch at LR 1e-4) val AUC 0.814708.
- 2026-10-06: M7 `both16` (PLE bins next to the bucketized embeddings; seed 2022; plateau LR rule; queue 1 finished 22:10 UTC, all 5
  jobs exit 0): valid 0.814708, test 0.814994, LL 0.437286, best epoch 6 of 8, 21,038,849 params. vs baseline (0.813972 /
  0.438288): +0.00102 test AUC, -0.00100 LogLoss; vs `ple16` (0.814703 / 0.437709): +0.00029 / -0.00042; vs BARS
  (0.814514 / 0.437631): +0.00048 test AUC and -0.00035 LogLoss, i.e. above BARS on both test metrics for this seed.
  The plateau rule dropped the LR after epoch 5. Single seed (2022, the best baseline seed by validation), LR-drop timing
  noise ~0.001: not a leaderboard claim; untested with the fixed epoch-4 drop (lrdrop4 alone: 0.814925 / 0.437186).
  Queue 1 summary, test AUC / LL (baseline 0.813972 / 0.438288): scalarlens_q 0.813533 / 0.439017, scalarlens 0.813898 /
  0.438118, multihash1m 0.813709 / 0.438556 (-32% params), ple16 0.814703 / 0.437709, both16 0.814994 / 0.437286.
  Pod idle 22:10-22:27 UTC before queue 2 (regularized ScalarLens) was launched.
- 2026-10-06: M9 `scalarlens_q_l2` (ScalarLens, quantile-initialized boundaries, L2 1e-4 on the ScalarLens parameters except the
  boundary logits; seed 2022; plateau LR rule; no fixed LR drop): valid 0.815155, test 0.815478, test LogLoss 0.436839, best
  epoch 9 of 11 (the plateau rule dropped the LR after epoch 8), 20,417,425 params. vs `scalarlens_q` without the regularizer
  (0.813257 / 0.813533 / 0.439017): +0.00195 test AUC, -0.00218 LogLoss, i.e. the L2 penalty removed the overfitting; vs baseline
  (0.813972 / 0.438288): +0.00151 AUC, -0.00145 LogLoss; vs BARS (0.814514 / 0.437631): +0.00096 AUC, -0.00079 LogLoss; vs `both16`
  (0.814994 / 0.437286): +0.00048 / -0.00045; best valid and test numbers of the session so far. The value 1e-4 was one a-priori
  choice (not tuned), seed 2022 is the baseline's best-validation seed (a tough seed for the comparison, but still one seed);
  plateau-rule LR timing noise ~0.001. Untested: with the fixed epoch-4 LR drop, with PLE bins, other seeds. `scalarlens_q_drop`
  (token dropout 0.1) started 23:15 UTC.
- 2026-10-07: M9 `scalarlens_q_drop` (ScalarLens, quantile boundaries, whole-token dropout 0.1, no L2; seed 2022; plateau LR rule):
  valid 0.814212, test 0.814662, test LogLoss 0.437674, best epoch 9 of 11 (LR dropped after epoch 8), 20,417,425 params, queue 2
  finished 00:00 UTC, both jobs exit 0. vs unregularized `scalarlens_q` (0.813257 / 0.813533 / 0.439017): +0.00113 test AUC,
  -0.00134 LogLoss; vs `scalarlens_q_l2` (0.815155 / 0.815478 / 0.436839): -0.00082 / +0.00084 (L2 is the better regularizer);
  vs baseline (0.813972 / 0.438288): +0.00069 / -0.00061; vs BARS (0.814514 / 0.437631): +0.00015 / -0.00006. Single seed, not tuned.
- 2026-10-07 morning summary (night run, seed 2022 only, plateau LR rule, test AUC / LogLoss; baseline 0.813972 / 0.438288,
  BARS 0.814514 / 0.437631). Queue 1: scalarlens_q 0.813533 / 0.439017, scalarlens 0.813898 / 0.438118, multihash1m 0.813709 /
  0.438556 (-32% params), ple16 0.814703 / 0.437709, both16 0.814994 / 0.437286. Queue 2: scalarlens_q_l2 0.815478 / 0.436839
  (best of the session, +0.00096 AUC over BARS), scalarlens_q_drop 0.814662 / 0.437674. Caveats: one seed, a-priori L2/dropout
  values, LR-drop timing noise ~0.001, not a leaderboard claim. Untested: ScalarLens+L2 with the epoch-4 fixed LR drop and with
  PLE bins, other seeds. Pod zg2efby39dz5ra deleted 2026-10-07 ~00:08 UTC; cost since 15:12 UTC on 10-06 about 9 h x $0.74/h = ~$6.7.
- 2026-10-07: M10 step 1, latency harness (`uv run python -m bars_dcn.bench.latency model.onnx requests.parquet --threads 1 4`;
  model from `scripts/export_onnx.py`, 1-epoch ScalarLens on full criteo so vocabularies are real-size: 20.4M params, 94 MB
  ONNX, 237 nodes). M2 Max, onnxruntime 1.30 CPU, batch 1, preloaded session, p50 / p99: 1 thread 529 / 607 us, 4 threads
  253 / 306 us. Measured through ORT: 49 GB/s streaming (1 thread), 113 GB/s (4), 109 / 412 GFLOP/s GEMM. The dense layers
  (23.4 MB weights, 11.7 MFLOP/row, five 1000-wide MLP layers + 3 cross layers) are memory bound at batch 1: bound 475 us
  (1 thread) = 90% of p50, 206 us (4 threads) = 81%. So at batch 1 the lever is bytes read (fp16/int8, narrower MLP,
  low-rank cross), not FLOPs. String lookups (26 `LabelEncoder`s) are ~4% of the profiled time, so lookup tricks are not worth
  it at this size; glue ops (Concat/Where/Mul/Cast/...) ~20% and dominate at 4 threads (per-node overhead). The ORT profiler
  inflates small nodes (kernel sum 665 us vs 529 us wall). Earlier BARS-bucket number in ARCHITECTURE.md: 0.19 ms/row.
- 2026-10-07: M10 step 2, int8 dense layers (`bars_dcn/onnx/quantize.py`: each Gemm with >= 100k weights becomes
  DynamicQuantizeLinear + MatMulInteger with per-channel int8 weights, i.e. 8 layers: 5 MLP, 3 cross; ORT's own
  `quantize_dynamic` silently did nothing for the torch-exported named Gemm nodes, hence the explicit rewrite). Same 1-epoch
  ScalarLens model, M2 Max, batch 1, preloaded, 1 thread: p50 534 -> 230 us (2.3x), p99 609 -> 262; dense weights 23.4 -> 6.0 MB,
  ONNX 94 -> 77 MB (the 80 MB embedding table is now most of the file). 4 threads: 253 -> 171 us. Accuracy on the first 300k
  test rows (paired, ORT): AUC 0.809481 -> 0.809454 (-0.00003), LogLoss 0.438657 -> 0.438684 (+0.00003): negligible (the absolute
  AUC is low because this model trained 1 epoch). The bound shown for int8 uses the fp32 GEMM peak, so its compute side is
  pessimistic. Remaining at 230 us: dense ~55%, the rest is glue ops / lookups / ScalarLens.
- 2026-10-07: M10 step 3, glue ops (fp32 graph 254 -> 145 nodes; behaviour unchanged). (a) ScalarLens readout: the 13 per-field MLPs are
  now stacked parameters (`readout_w1/b1/w2/b2`, one batched einsum; initialized from the same per-field `nn.Linear` draws, output
  identical to the old module within 2e-8 on a seeded input; old checkpoints do not load). (b) `MissingFiller` emits one
  Concat -> IsNaN -> Where -> Split over all columns, the estimator concatenates the numeric block first and casts it once, and a graph
  pass (`onnx/_simplify.py`, run in `to_onnx`) cancels `Concat(Split(x))`. Re-exported 1-epoch ScalarLens (a new training run, so its AUC differs
  from step 2's), M2 Max, batch 1, p50: fp32 527 us (1 thread) / 234 (4); int8 211 us (1) / 160 (4), vs step 2 int8 230 / 171
  (-8% / -6%). int8 vs fp32 AUC on 300k test rows 0.809172 vs 0.809166 (+0.00001), LogLoss 0.438907 vs 0.438908. Left at 211 us: dense
  ~55-60% (bound 125 us), string lookups ~26 profiled us, the rest is ScalarLens coordinate/normalization ops (einsum, slices, mul/add);
  diminishing returns without restructuring ScalarLens. fp32 is unchanged because it is memory bound.
- 2026-10-07: M10 step 4, embedding table (`quantize_embedding`, int8 with a scale per row, or fp16; the table is 909,700 x 16 = 58 MB fp32).
  M2 Max, batch 1, p50 1 thread / 4 threads, ONNX size, AUC on 300k test rows (same 1-epoch model as step 3):
  fp32 535 / 242 us, 94 MB, 0.809166; fp32 + emb int8 540 / 238, 54 MB, 0.809166; int8 dense 204 / 160, 77 MB, 0.809172;
  int8 dense + emb fp16 218 / 151, 48 MB, 0.809173; int8 dense + emb int8 212 / 145, 37 MB, 0.809175 (LogLoss 0.438902 vs 0.438908).
  So the table quantization costs no accuracy and shrinks the file by 52% (77 -> 37 MB), but does not speed up batch 1 on this
  machine: a request gathers 26 random rows of 64 B (one cache line each), which is latency not bandwidth bound, and the 4-thread gain
  (160 -> 145 us) is the only visible one. It should matter on x86 if the 15-18 MB int8 table fits a shared L3 where 58 MB does
  not, and for reload time / RAM per model. To be re-measured on Hetzner; the realistic access pattern is skewed (hot rows) while the
  bench replays 20k distinct test rows.
- 2026-10-07: M10 step 5, x86 (Hetzner `ccx23`, fsn1, ~0.167 EUR/h, created and deleted the same day: AMD EPYC-Milan (Zen 3), 4 vCPU = 2 cores x
  2 SMT threads, AVX2 only (no AVX-512 / VNNI), L2 1 MiB per core, L3 32 MiB, KVM; Python 3.12, onnxruntime 1.30; the bench is
  run as `python latency.py` there because `bars_dcn.bench/__init__` imports torch). Same models as the M2 runs (step 3/4), batch 1,
  preloaded, p50 (p99):
  fp32 497 us (613) 1 thread, 332 us 2 threads, 343 us 4; int8 dense 210 us (232) 1 thread, 193 us 2, 271 us 4; int8 + emb int8 215 / 197 / 270;
  int8 + emb fp16 209 / 191 / 270. A second fp32 run measured 628 us (1 thread), so this VM has ~25% run-to-run noise; int8 repeated at 210-220.
  AUC on 300k test rows on x86 (ORT AVX2 u8s8 kernels): fp32 0.809166, int8 0.809160 (-0.000006), int8 + emb int8 0.809163. Findings:
  (1) int8 is 2.3-2.9x faster at 1 thread even without VNNI, and the accuracy cost is nil; (2) threads: 4 vCPUs (2 cores) are slower than 2, 2 barely
  beats 1 for int8: serve with 1 thread per request (and scale by replicas); (3) embedding quantization gives no latency gain on x86 either
  (the table does not dominate: 26 gathered rows), only the file size; (4) the first roofline used a DRAM-sized buffer, so the fp32 "bound" (639 us)
  exceeded the measured time, as the 23 MB of weights live in the 32 MB L3; the probe now streams a buffer the size of the model's dense weights
  (22 MB: 47 GB/s on 1 thread, 6 MB: 62 GB/s). With it, int8 dense is ~108 us of the 220 us (49%), fp32 dense 78% of p50. The other half of int8 is
  glue: on x86 the 26 `LabelEncoder` string lookups are ~12% of the profile (more than on the M2), then Mul/Add/Einsum (ScalarLens), Concat, Reshape.
  Next levers: ScalarLens restructuring (fewer small ops) and cheaper categorical lookups (precomputed hash / one fused lookup).
- 2026-10-07: M10 step 6, ScalarLens graph. `coordinate` now picks the interval's boundaries and knot vectors with indexed lookups (flattened
  tables + per-field offsets) instead of one-hot `Equal/Cast/Mul/ReduceSum` and two einsums. Same math: output equals the previous module within
  2e-8 on a seeded input; `state_dict` unchanged (the offsets are a non-persistent buffer); training step on MPS (batch 10k, module alone) 21.0 -> 18.7 ms.
  ScalarLens alone in ORT (13 numerics, 26 categorical tokens, batch 1, 1 thread, M2 Max): 81 -> 77 nodes, p50 43 -> 37 us, i.e. ~6 us of the ~200 us
  request (3%, inside the end-to-end noise, so the full model was not re-exported). Tried and dropped: a paired (start,end)/(lower,upper) lookup (2 gathers
  instead of 4) measured 40 us, not better. What is left is the 3-step recurrence (about 10 small nodes per step) and the head/readout einsums: ~35 us, ScalarLens
  is ~17% of the request, so restructuring it further can give at most ~10 us; not pursued.
- 2026-10-07: M10 step 7, one lookup for all string columns. `OrdinalEncoder` on strings is emitted as Concat of the columns -> `StringConcat` with a
  "<column number>:" prefix -> a single `LabelEncoder` over the prefixed vocabularies (909k keys) -> Split back into columns (cancelled against
  the estimator's Concat by `simplify`); semantics unchanged (null / "" / unseen -> OOV 0, covered by the e2e edge-row tests). Integer
  columns keep one `LabelEncoder` each. ONNX opset 18 -> 20 (StringConcat). Where the time was, M2 Max 1 thread, 26 string inputs: the lookups-only graph ran
  31 us, of which 16.5 us is ORT's binding of 26 numpy string arrays (an identity graph on the same inputs, a client cost that no graph change removes); the
  26 lookups were ~15 us and the fused one ~3 us (19.6 us total). End to end (re-exported 1-epoch ScalarLens, batch 1, 1 thread, also including the
  step 6 ScalarLens change): fp32 487 us, int8 201 us (v2: 211 us, -10 us / -5%), graph 145 -> 117 nodes (fp32); AUC on 300k test rows int8 0.809062 vs fp32 0.809068
  (a fresh 1-epoch run again: absolute AUC varies between runs, the int8-fp32 gap does not). The x86 gain of this step is not yet measured (the LabelEncoder
  share is larger there, ~12% of the profile).
- 2026-10-07: M10 side check, newer opset. ScalarLens alone, batch 1, 1 thread, M2: replacing the manual RMS norm by `F.rms_norm` and exporting at opset 23
  (RMSNormalization) gives 77 -> 74 nodes and 37.3 -> 37.1 us: no gain (ORT already fuses this pattern, and kernels do not depend on the opset). Opset stays 20
  (the minimum for StringConcat); reverted.
- 2026-10-07: M10 stage 1, MLP speed sweep (`scripts/speed_sweep.py`: untrained variants, real vocabularies, int8 dense layers; latency only, accuracy is stage 2).
  Also in this step: BatchNorm folded into the preceding Linear at export (`fold_batch_norm`, exact; no measurable speed change because ORT already fused it). p50, 1 thread, batch 1:
  M2 Max int8 / fp32 us: base [1000]x5 198 / 495; mlp1000x2 139 / 260; mlp512x3 128 / 198; mlp256x3 117 / 168; dim8 (embedding_dim 8) 186 / 395;
  dim8_mlp512x3 113 / 143; mix (low-rank mixture cross, rank 32, 4 experts) 372 / 601; mlp512x3_mix 294 / 330; dim8_mlp512x3_mix 212 / 235.
  Hetzner ccx23 (a new VM, ~35% slower than the first one: the same v2 graph measured 289 us now vs 210 us before, so only same-session numbers compare), int8 + int8 table:
  base 270; mlp1000x2 203; mlp512x3 194; mlp256x3 184; dim8 251; dim8_mlp512x3 183; mix 462; fp32 base 694-739. Same host, same session: the step 7 fused lookup graph (v3) is 270 vs 289 us
  for v2 (-19 us, -6.5%), so the fused lookup helps on x86 as expected. Findings: (1) the low-rank *mixture* cross is SLOWER than full-rank (372 vs 198 us on M2, 462 vs 270 on x86)
  despite 20% fewer weights: its experts, gating and tanh are many tiny ops; a plain single low-rank cross (two matmuls) would be needed to gain; (2) narrowing the MLP gives -25..-32% on x86
  and flattens out: mlp512x3 194 vs mlp256x3 184 vs dim8_mlp512x3 183 us, the floor (lookups, ScalarLens, glue, client binding) is ~170-180 us on that host (~100 us on the M2);
  (3) embedding_dim 8 alone is a small gain (-7%) but combined with a 512x3 MLP reaches the floor; (4) in stage 2, candidates to train: mlp1000x2, mlp512x3, mlp256x3, dim8_mlp512x3
  (plus the known base 0.815478 as reference); the mixture variants are dropped. dim8_mlp256x3_mix had no layer large enough to quantize and was not timed.
- 2026-10-07: Stopped here by the user: cleanup and README. Deleted `runs/latency` (2.1 GB, regenerable with `scripts/export_onnx.py` / `scripts/speed_sweep.py`), no cloud resources left
  (RunPod pods: none, Hetzner servers: none). M10 stage 2 (train mlp1000x2 / mlp512x3 / mlp256x3 / dim8_mlp512x3, fixed epoch-4 LR drop or plateau rule, AUC vs latency) is not started;
  open questions for it: the AUC budget, the LR schedule, whether to add a LayerNorm-in-MLP variant (cannot be folded away, so it can only cost latency).
