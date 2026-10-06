"""Benchmark runner: BARS datasets, TOML configs, one ``metrics.json`` per seed."""

from bars_dcn.bench.runner import apply_overrides, load_config, prepare, run_seed

__all__ = ["apply_overrides", "load_config", "prepare", "run_seed"]
