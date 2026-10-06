# CLAUDE.md

## Why

Build a production-quality, standalone implementation of **DCNv2** (Deep & Cross Network v2,
https://arxiv.org/abs/2008.13535) and use it to **beat the existing DCNv2 results on the
[BARS-CTR leaderboard](https://openbenchmark.github.io/BARS/CTR/index.html)**.

- BARS-CTR is the open benchmark for CTR prediction. Its reference DCNv2 numbers come from
  FuxiCTR. Our target is a leaderboard entry that ranks higher on test AUC / LogLoss.
- `../scikit-rank` is the **reference implementation only**. It already has DCNv2 and a BARS
  runner (`src/scikit_rank/modules/dcn.py`, `src/scikit_rank/sklearn/dcn.py`, `exps/bars/`).
  It shows what works (FuxiCTR parity config, PLE numeric encoder, multihash embeddings,
  5-seed protocol). This repo does **not** depend on it and does not copy its scope.

## What

A small, DCNv2-only codebase: model, training loop, data pipeline, and a config-driven
benchmark runner.

In scope:
- DCNv2 model (cross network + deep network, stacked and parallel structures, low-rank
  cross / mixture of experts as options).
- Training and evaluation that follow the BARS protocol: fixed pre-split data, MD5
  verification of splits, model selection by validation AUC with early stopping, test
  ROC-AUC and LogLoss, multiple fixed seeds.
- Benchmark datasets: BARS-CTR `criteo_x4` and `avazu_x4`.
- Polars `LazyFrame` input end to end, so the full x4 splits never have to sit in memory as
  raw strings.
- Reproducible runs: one config per (model x dataset), results stored as `metrics.json`
  per seed.

Out of scope (unless explicitly added later):
- Other model families (FinalMLP, FinalNet, TabM, DESTINE, GBDT baselines).
- Ranking losses, multi-GPU/DDP.
- Reproducing other leaderboard entries.

## Architecture

Preprocessing, model and postprocessing are separate sklearn components composed in a
`Pipeline`, and the full pipeline must export to a single ONNX graph. Details, constraints
and rationale: [ARCHITECTURE.md](ARCHITECTURE.md). Read it before changing any model,
transformer or export code.

End-to-end tests run on a small committed `criteo_x4` sample (see "End-to-end test data" in
ARCHITECTURE.md); benchmark numbers come only from the full splits.

## Commands

```bash
uv sync                      # install (Python 3.14)
uv run pytest                # tests
uv run ruff check            # lint (select ALL, see pyproject.toml); never auto-fixes
uv run ruff format           # format
uv run ty check               # type check (Astral's ty; keep it clean)
uv run python scripts/make_sample.py [--dataset avazu_x4]   # regenerate an e2e sample (needs data/<Dataset>/)
uv run pytest -m full_data    # opt-in: BARS preprocessing parity on the full train split (needs data/Criteo_x4/, ~20 s)
uv run python -m bars_dcn.bench configs/criteo_x4_dcnv2.toml --seeds 2019   # full run on MPS -> runs/
uv run tensorboard --logdir runs   # epoch metrics + signal-propagation plots
```

Always use `uv run`; don't activate the venv by hand.

## Conventions for sklearn components

Every estimator/transformer (anything subclassing `BaseEstimator`):

- Real imports for annotation names. sklearn's `get_params`/`set_params`/`clone`/`repr` and
  `get_metadata_routing` call `inspect.signature`, which evaluates annotations at runtime on
  3.14. Never move an annotation-only import under `TYPE_CHECKING` in these files (ruff's `TC`
  rules are disabled for `estimator.py` and `preprocessing/`; put new components there).
- Declare `__sklearn_tags__` and honor it in behavior (e.g. `allow_nan=False` means NaN raises).
- Register a `Case` in `tests/test_sklearn_contract.py`; a guard test fails otherwise.
- `__init__` only stores parameters; `fit` sets only attributes ending in `_`.

## Progress

Track status in [PROGRESS.md](PROGRESS.md): update it when a task starts, finishes or
changes scope.

## Success criteria

- Parity first: our DCNv2 matches the published BARS/FuxiCTR DCNv2 numbers within seed noise
  on each dataset. Until that holds, improvements are not trusted.
- Then: exceed the best existing DCNv2 entry on each dataset's test AUC, over the same
  number of seeds and the same evaluation protocol.

## Environment

- **Python 3.14** (`requires-python >=3.14`), managed with `uv`. Checked 2026-10-06: torch,
  polars, scikit-learn, skl2onnx, onnxruntime, onnx, accelerate, pyarrow and numpy all
  resolve to binary wheels on 3.14. If a future dependency lacks 3.14 support, say so
  before lowering the Python version.
- **Compute:** local Apple Silicon (MPS), with device placement via 🤗 Accelerate. Code must
  not assume CUDA. ONNX export runs from the CPU copy of the model.
- **Production = ONNX:** the deliverable that matters for serving is the exported pipeline
  (see ARCHITECTURE.md).
- **Data:** large datasets live under `/data/` (gitignored), never in git.

## Open questions

- "Beat the leaderboard": versus the DCNv2 row only, or the overall best model per dataset?
- Hyperparameter search budget, and whether any GPU beyond local MPS is available.
