"""Train and evaluate the educational recorded-label fraud baseline."""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DATA_PATH
from .training import DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIR, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = train_and_evaluate(args.csv, args.manifest, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
