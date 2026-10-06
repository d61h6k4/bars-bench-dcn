"""Every committed experiment queue file must parse, apply its overrides and build a pipeline."""

import shlex
from pathlib import Path

import pytest

from bars_dcn.bench.queue import _overrides
from bars_dcn.bench.runner import apply_overrides, build_pipeline, load_config

ROOT = Path(__file__).parents[2]
LINES = [
    pytest.param(path.name, line, id=f"{path.name}:{index}")
    for path in sorted((ROOT / "experiments").glob("*.txt"))
    for index, line in enumerate(path.read_text().splitlines())
    if line.strip() and not line.startswith("#")
]


@pytest.mark.parametrize(("file", "line"), LINES)
def test_queue_line_builds_its_pipeline(file, line):
    arguments = shlex.split(line)  # as the queue does: quotes inside --set values must survive
    config = apply_overrides(load_config(ROOT / arguments[0]), _overrides(arguments))
    assert build_pipeline(config, seed=0) is not None, file
