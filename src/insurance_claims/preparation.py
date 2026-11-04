"""Create a reproducible fraud dataset split and fit preprocessing on train only."""

import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import (
    DEFAULT_DATA_PATH,
    EXCLUDED_ALWAYS,
    FRAUD_AVAILABILITY_REVIEW_FIELDS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET,
)
from .data import load_dataset
from .preprocessing import build_preprocessor, engineer_features, prepare_features, split_dataset

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_counts(values: pd.Series) -> dict[str, int]:
    return {str(label): int(count) for label, count in sorted(Counter(values).items(), key=lambda pair: str(pair[0]))}


def _validate_target(values: pd.Series) -> None:
    """Require an integer target that contains only the configured labels."""
    if values.isna().any():
        raise ValueError(f"Target {TARGET!r} contains missing values.")
    if not pd.api.types.is_integer_dtype(values):
        raise ValueError(f"Target {TARGET!r} must be integer 0/1, found dtype {values.dtype}.")
    unexpected = set(values.unique().tolist()) - {NEGATIVE_LABEL, POSITIVE_LABEL}
    if unexpected:
        raise ValueError(f"Target {TARGET!r} contains unexpected labels: {sorted(unexpected)}")


def prepare_fraud_dataset(
    csv_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Write a split manifest and a preprocessing artifact; fit on train only.

    Validation and test features are split and indexed but never passed to the
    preprocessor's fit or transform methods.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    frame = engineer_features(load_dataset(csv_path))
    if TARGET not in frame.columns:
        raise ValueError(f"Required fraud target {TARGET!r} is missing.")
    _validate_target(frame[TARGET])

    target_counts = _class_counts(frame[TARGET])
    X, y = prepare_features(frame, TARGET)
    split = split_dataset(X, y, random_state=seed)

    preprocessor = build_preprocessor(split.X_train)
    preprocessor.fit(split.X_train)

    partitions = {
        "train": (split.X_train, split.y_train),
        "validation": (split.X_validation, split.y_validation),
        "test": (split.X_test, split.y_test),
    }
    split_manifest = {}
    for name, (features, labels) in partitions.items():
        split_manifest[name] = {
            "row_count": int(len(features)),
            "source_row_indices": [int(index) for index in features.index],
            "target_counts": _class_counts(labels),
        }

    review_fields = [
        {
            "column": column,
            "reason": "Claim timing is undocumented; confirm availability at the intended prediction moment.",
        }
        for column in FRAUD_AVAILABILITY_REVIEW_FIELDS
        if column in X.columns
    ]
    excluded_columns = [column for column in frame.columns if column in EXCLUDED_ALWAYS]
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

    manifest: dict[str, Any] = {
        "source_csv": csv_path.as_posix(),
        "source_sha256": _sha256(csv_path),
        "row_count": int(len(frame)),
        "target": TARGET,
        "target_dtype": str(frame[TARGET].dtype),
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "target_counts": target_counts,
        "target_values_observed": list(target_counts),
        "seed": int(seed),
        "split_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_method": f"two-stage random split stratified on {TARGET}",
        "partitions": split_manifest,
        "excluded_columns": excluded_columns,
        "input_feature_columns": list(X.columns),
        "availability_review_fields": review_fields,
        "preprocessing": {
            "fit_partition": "train",
            "validation_transformed": False,
            "test_transformed": False,
            "numeric": "median imputation then standard scaling",
            "ordinal": "mapped to integers in the configured order; unknown labels become missing; median imputation then standard scaling",
            "categorical": "most-frequent imputation then ordinal encoding; unknown categories become -1",
            "transformed_feature_count": len(feature_names),
        },
        "prediction_model_trained": False,
        "software_versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "joblib")
        },
        "notes": [
            f"The target is the observed {TARGET} column; label semantics are not independently documented.",
            "The split is reproducible from the source row order, seed, and software dependencies.",
            "Identifier and demographic columns listed in excluded_columns are not model inputs.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "fraud_split_manifest.json"
    artifact_path = output_dir / "fraud_preparation.joblib"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    joblib.dump(
        {
            "target": TARGET,
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "seed": int(seed),
            "source_csv": csv_path.as_posix(),
            "source_sha256": manifest["source_sha256"],
            "software_versions": manifest["software_versions"],
            "input_feature_columns": list(X.columns),
            "excluded_columns": excluded_columns,
            "preprocessor": preprocessor,
            "transformed_feature_names": feature_names,
            "partitions": {
                name: {
                    "source_row_indices": split_manifest[name]["source_row_indices"],
                    "target_counts": split_manifest[name]["target_counts"],
                }
                for name in partitions
            },
        },
        artifact_path,
    )
    manifest["saved_manifest"] = manifest_path.as_posix()
    manifest["saved_preprocessing_artifact"] = artifact_path.as_posix()
    return manifest
