"""Rank exploratory fraud-label models on the saved validation partition only."""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DATA_PATH
from .feature_set_comparison import (
    DEFAULT_BASELINE_MODEL,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    rank_models_on_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = rank_models_on_validation(
        args.csv, args.manifest, args.baseline_model, args.output_dir
    )
    print(json.dumps({
        "status": report["status"],
        "model_ranking": report["model_ranking"],
        "models": report["models"],
        "highest_validation_average_precision_model": report["highest_validation_average_precision_model"],
        "winner_validation_metrics": report["winner_validation_metrics"],
        "model_artifacts": report["model_artifacts"],
        "test_partition_used_for_fit_selection_threshold_or_metrics": report["test_partition_used_for_fit_selection_threshold_or_metrics"],
        "report_path": report["report_path"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
