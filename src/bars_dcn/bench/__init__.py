"""Benchmark runner: BARS datasets, TOML configs, one ``metrics.json`` per seed."""

from bars_dcn.bench.runner import load_config, run_seed

__all__ = ["load_config", "run_seed"]
