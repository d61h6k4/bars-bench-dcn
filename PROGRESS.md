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
  MLDCN layer), and other tricks found while reading. Each needs ONNX exportability checked.
- [ ] **M9 Compare with ScalarLens:** arXiv 2609.29182 (numerical embeddings for CTR). Read the
  paper first; then compare against our M5/M7 numerics (standardize / bucketize / PLE) under the
  same protocol, if it can be exported to ONNX. "Beat the leaderboard" scope stays open.

## Next actions

- [ ] Add an AUC-minus-logloss monitor option (BARS avazu monitors `{AUC: 1, logloss: -1}`)
- [ ] M8: fixed-epoch LR schedule vs plateau trigger (the criteo gap tracks LR-drop timing)
- [ ] Download `avazu_x4` (need link + reference MD5s)

## Open questions

- "Beat the leaderboard": DCNv2 row only, or best model per dataset?
- ScalarLens (M9): does the paper report on BARS splits / ship code, and can its embedding be exported to ONNX? Not read yet.
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
