"""``uv run python -m bars_dcn.bench configs/criteo_x4_dcnv2.toml --seeds 2019``."""

import argparse
import logging
from pathlib import Path

from bars_dcn.bench.runner import load_config, run_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2019])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(args.config)
    out_dir = args.out_dir or Path("runs") / args.config.stem
    for seed in args.seeds:
        run_seed(config, seed, out_dir, args.data_root)


if __name__ == "__main__":
    main()
