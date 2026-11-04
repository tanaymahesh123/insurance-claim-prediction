"""Train and compare type-specific exploratory fraud-label models."""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DATA_PATH
from .incident_specialists import DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIR, train_incident_specialists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = train_incident_specialists(args.csv, args.manifest, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "specialists": report["specialists"],
        "test_partition_used_for_fit_selection_threshold_or_metrics": report[
            "test_partition_used_for_fit_selection_threshold_or_metrics"
        ],
        "artifacts": report["artifacts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
