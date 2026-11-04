"""Run read-only data quality checks on the project's raw CSV."""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DATA_PATH
from .data import load_dataset
from .quality import inspect_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Raw CSV path (default: {DEFAULT_DATA_PATH.as_posix()})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report; the raw CSV is never modified.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_dataset(args.csv)
    report = inspect_dataset(frame, args.csv)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
