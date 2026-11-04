"""Run bounded specialist error-pattern experiments on train/validation only."""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DATA_PATH
from .incident_specialists import DEFAULT_MANIFEST
from .specialist_error_experiments import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPECIALISTS,
    run_specialist_error_experiments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--specialists", type=Path, default=DEFAULT_SPECIALISTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run_specialist_error_experiments(
        args.csv, args.manifest, args.specialists, args.output_dir
    )
    print(json.dumps({
        "status": report["status"],
        "specialists": report["specialists"],
        "test_used_for_fit_model_selection_threshold_or_metrics": report["partitions_used"][
            "test_used_for_fit_model_selection_threshold_or_metrics"
        ],
        "artifacts": report["artifacts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
