"""``uv run python -m bars_dcn.bench configs/criteo_x4_dcnv2.toml --seeds 2019 --name baseline``."""

import argparse
import logging
from pathlib import Path

from bars_dcn.bench.runner import apply_overrides, load_config, prepare, run_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2019])
    parser.add_argument("--name", help="run name; results go to runs/<name> (default: config stem)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="section.key=value",
        help="override a config value (TOML syntax), e.g. --set model.lr_drop_epochs=[6]",
    )  # fmt: skip
    parser.add_argument("--prepare-only", action="store_true", help="only build the parquet files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = apply_overrides(load_config(args.config), args.overrides)
    if args.prepare_only:
        prepare(config, args.data_root)
        return
    out_dir = args.out_dir or Path("runs") / (args.name or args.config.stem)
    for seed in args.seeds:
        run_seed(config, seed, out_dir, args.data_root)


if __name__ == "__main__":
    main()
