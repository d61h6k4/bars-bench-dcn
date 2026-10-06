"""Run a file of benchmark commands in parallel processes (e.g. several ablations on one GPU).

Each non-comment line holds the arguments of ``python -m bars_dcn.bench``. Data is prepared once
up front, every child's full output goes to ``runs/logs/``, and only the progress and result lines
are echoed (prefixed with the run name), so the queue's own output stays short.

    uv run python -m bars_dcn.bench.queue experiments/m8_lr_schedule.txt --parallel 4 --device auto
"""

import argparse
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bars_dcn.bench.runner import apply_overrides, load_config, prepare

SHOWN = re.compile(r"epoch \d+:|RESULT|seed \d+:|Error|Traceback")


def _name(arguments: list[str], index: int) -> str:
    return arguments[arguments.index("--name") + 1] if "--name" in arguments else f"job{index}"


def _overrides(arguments: list[str]) -> list[str]:
    return [arguments[i + 1] for i, argument in enumerate(arguments) if argument == "--set"]


def _run(index: int, arguments: list[str], data_root: Path, device: str | None) -> tuple[str, int]:
    name = _name(arguments, index)
    command = [sys.executable, "-m", "bars_dcn.bench", *arguments, "--data-root", str(data_root)]
    if device:
        command += ["--set", f'model.device="{device}"']
    log = Path("runs/logs") / f"{index:02d}_{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as sink:
        process = subprocess.Popen(  # noqa: S603
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None  # noqa: S101
        for line in process.stdout:
            sink.write(line)
            if SHOWN.search(line):
                print(f"[{name}] {line.rstrip()}", flush=True)  # noqa: T201
    return name, process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--device", default=None, help="override model.device for every job")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    jobs = [
        shlex.split(line)
        for line in args.experiments.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for config_path in sorted({Path(job[0]) for job in jobs}):
        job_overrides = [
            j[i + 1]
            for j in jobs
            if Path(j[0]) == config_path
            for i, a in enumerate(j)
            if a == "--set"
        ]
        apply_overrides(load_config(config_path), job_overrides)  # fail fast on a bad override
        prepare(load_config(config_path), args.data_root)
    print(f"QUEUE_START {len(jobs)} jobs, {args.parallel} in parallel", flush=True)  # noqa: T201
    with ThreadPoolExecutor(args.parallel) as pool:
        futures = [
            pool.submit(_run, index, job, args.data_root, args.device)
            for index, job in enumerate(jobs)
        ]
        results = [future.result() for future in futures]
    for name, code in results:
        print(f"QUEUE_JOB {name} exit={code}", flush=True)  # noqa: T201
    print("QUEUE_DONE", flush=True)  # noqa: T201
    sys.exit(any(code for _, code in results))


if __name__ == "__main__":
    main()
