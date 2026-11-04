"""Prepare fraud train/validation/test splits without fitting a predictor."""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DATA_PATH
from .preparation import DEFAULT_OUTPUT_DIR, DEFAULT_SEED, prepare_fraud_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    result = prepare_fraud_dataset(args.csv, args.output_dir, args.seed)
    summary = {
        "target": result["target"],
        "target_counts": result["target_counts"],
        "partitions": {
            name: {
                "row_count": details["row_count"],
                "target_counts": details["target_counts"],
            }
            for name, details in result["partitions"].items()
        },
        "excluded_columns": result["excluded_columns"],
        "availability_review_fields": [
            item["column"] for item in result["availability_review_fields"]
        ],
        "preprocessing_fit_partition": result["preprocessing"]["fit_partition"],
        "prediction_model_trained": result["prediction_model_trained"],
        "saved_manifest": result["saved_manifest"],
        "saved_preprocessing_artifact": result["saved_preprocessing_artifact"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
