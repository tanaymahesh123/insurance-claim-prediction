"""Run an exploratory fraud-label model comparison on train and validation only."""

import argparse
import json
from pathlib import Path

from .comparison import (
    DEFAULT_BASELINE_MODEL,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    compare_on_validation,
)
from .config import DEFAULT_DATA_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = compare_on_validation(
        args.csv, args.manifest, args.baseline_model, args.output_dir
    )
    summary = {
        "status": report["status"],
        "target": report["target"],
        "models": {
            name: {
                "features": details["features"],
                "validation_metrics": details["validation_metrics"],
            }
            for name, details in report["models"].items()
        },
        "selected_model_by_validation_only": report["selected_model_by_validation_only"],
        "test_evaluated": report["test_evaluated"],
        "artifacts": report["artifacts"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
