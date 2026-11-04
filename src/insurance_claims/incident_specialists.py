"""Train bounded, exploratory Random Forest specialists by vehicle category.

Only the saved train and validation assignments are used. The test partition is
not read for fitting, feature-group choice, threshold choice, or reporting.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .config import (
    DEFAULT_DATA_PATH,
    EXCLUDED_ALWAYS,
    FEATURES,
    FRAUD_AVAILABILITY_REVIEW_FIELDS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET,
)
from .data import load_dataset
from .metrics import checked_metric
from .preprocessing import build_preprocessor, classify_columns, engineer_features

ROUTING_FIELD = "VehicleCategory"
DEFAULT_MANIFEST = Path("data/processed/fraud_split_manifest.json")
DEFAULT_OUTPUT_DIR = Path("data/processed")
MODEL_FILENAME = "fraud_vehicle_specialists_v1.joblib"
REPORT_FILENAME = "fraud_vehicle_specialists_v1.json"

# The model family and settings are fixed for all categories and groups.
RF_SETTINGS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "class_weight": "balanced_subsample",
}

FORBIDDEN = set(EXCLUDED_ALWAYS) | {TARGET, ROUTING_FIELD}

# VehicleCategory only chooses the specialist, so it is never a model input. Each
# category compares the reference group against the same two ablations.
FULL_FEATURES = tuple(feature for feature in FEATURES if feature not in FORBIDDEN)
CALENDAR_FIELDS = (
    "Month", "WeekOfMonth", "DayOfWeek", "MonthClaimed",
    "WeekOfMonthClaimed", "DayOfWeekClaimed", "Year",
)
FEATURE_GROUPS = {
    "full": FULL_FEATURES,
    "no_claim_timing": tuple(
        feature for feature in FULL_FEATURES if feature not in FRAUD_AVAILABILITY_REVIEW_FIELDS
    ),
    "no_calendar": tuple(feature for feature in FULL_FEATURES if feature not in CALENDAR_FIELDS),
}
GROUP_RATIONALE = {
    "full": "Reference group: every configured input, with VehicleCategory reserved for routing.",
    "no_claim_timing": "Removes the claim-timing fields whose availability at prediction time is unverified.",
    "no_calendar": "Removes calendar fields to test whether the remaining policy and incident inputs hold up without them.",
}

# Require a modest number of examples in each class before presenting validation
# metrics. This is a run-safety floor, not a guarantee of statistical reliability.
MIN_TRAIN_PER_CLASS = 100
MIN_VALIDATION_PER_CLASS = 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(y: pd.Series) -> dict[str, int]:
    return {str(label): int((y == label).sum()) for label in (NEGATIVE_LABEL, POSITIVE_LABEL)}


def _scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    if POSITIVE_LABEL not in classes:
        raise ValueError(f"Training partition does not contain the {POSITIVE_LABEL!r} label.")
    return model.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]


def _choose_threshold(y: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, scores, pos_label=POSITIVE_LABEL)
    if not len(thresholds):
        raise ValueError("Validation data did not yield threshold candidates.")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    tied = np.flatnonzero(np.isclose(f1, f1.max()))
    chosen = int(tied[-1])  # highest threshold among maximum-F1 ties
    return float(thresholds[chosen]), float(f1[chosen])


def _metrics(y: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    actual = y.to_numpy() == POSITIVE_LABEL
    predicted_y = scores >= threshold
    return {
        "average_precision": checked_metric(average_precision_score(actual, scores)),
        "precision_Y": checked_metric(precision_score(actual, predicted_y, zero_division=0)),
        "recall_Y": checked_metric(recall_score(actual, predicted_y, zero_division=0)),
        "f1_Y": checked_metric(f1_score(actual, predicted_y, zero_division=0)),
        "selected_threshold_for_Y": float(threshold),
        "validation_rows_predicted_Y": int(predicted_y.sum()),
        "validation_rows_predicted_Y_percent": float(predicted_y.mean() * 100),
        "validation_false_positive_count": int((predicted_y & ~actual).sum()),
        "validation_rows": int(len(y)),
    }


def _validate_manifest(manifest: dict[str, Any], frame: pd.DataFrame, source_hash: str) -> tuple[list[int], list[int]]:
    if manifest.get("target") != TARGET:
        raise ValueError(f"Split manifest target must be {TARGET!r}.")
    if manifest.get("source_sha256") != source_hash:
        raise ValueError("CSV checksum differs from the source used to create the saved split.")
    parts = manifest["partitions"]
    train = [int(i) for i in parts["train"]["source_row_indices"]]
    validation = [int(i) for i in parts["validation"]["source_row_indices"]]
    test = set(int(i) for i in parts["test"]["source_row_indices"])
    if not train or not validation:
        raise ValueError("Saved training and validation partitions must be non-empty.")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("Saved training or validation indices contain duplicates.")
    if set(train) & set(validation) or (set(train) | set(validation)) & test:
        raise ValueError("Saved training, validation, and test indices must be disjoint.")
    if min(train + validation) < 0 or max(train + validation) >= len(frame):
        raise ValueError("A saved training or validation index is outside the CSV row range.")
    return train, validation


def _fit_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    seed: int,
) -> tuple[Pipeline, dict[str, float | int]]:
    model = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("classifier", RandomForestClassifier(**RF_SETTINGS, random_state=seed, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    validation_scores = _scores(model, X_validation)
    threshold, _ = _choose_threshold(y_validation, validation_scores)
    return model, _metrics(y_validation, validation_scores, threshold)


def train_incident_specialists(
    csv_path: Path = DEFAULT_DATA_PATH,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Compare feature groups per vehicle category and save each validation winner."""
    csv_path, manifest_path, output_dir = Path(csv_path), Path(manifest_path), Path(output_dir)
    for path in (csv_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hash = _sha256(csv_path)
    frame = engineer_features(load_dataset(csv_path))
    if TARGET not in frame or ROUTING_FIELD not in frame:
        raise ValueError(f"CSV is missing the target or {ROUTING_FIELD} routing field.")
    train_indices, validation_indices = _validate_manifest(manifest, frame, source_hash)
    train_frame = frame.iloc[train_indices].copy()
    validation_frame = frame.iloc[validation_indices].copy()
    categories = sorted(
        set(train_frame[ROUTING_FIELD].dropna().astype(str))
        | set(validation_frame[ROUTING_FIELD].dropna().astype(str))
    )
    if not categories:
        raise ValueError("No observed vehicle categories are available for specialist training.")

    models_by_category: dict[str, Any] = {}
    report_by_category: dict[str, Any] = {}
    skipped_categories: dict[str, Any] = {}
    seed = int(manifest["seed"])
    for category in categories:
        train_rows = train_frame[train_frame[ROUTING_FIELD].astype(str) == category]
        validation_rows = validation_frame[validation_frame[ROUTING_FIELD].astype(str) == category]
        y_train, y_validation = train_rows[TARGET], validation_rows[TARGET]
        train_counts, validation_counts = _counts(y_train), _counts(y_validation)
        if (
            min(train_counts.values()) < MIN_TRAIN_PER_CLASS
            or min(validation_counts.values()) < MIN_VALIDATION_PER_CLASS
        ):
            skipped_categories[category] = {
                "training_counts": train_counts,
                "validation_counts": validation_counts,
                "reason": (
                    f"Below the floor of {MIN_TRAIN_PER_CLASS} training or "
                    f"{MIN_VALIDATION_PER_CLASS} validation examples per class."
                ),
            }
            continue

        candidate_results: dict[str, Any] = {}
        candidate_models: dict[str, Pipeline] = {}
        for group_name, features in FEATURE_GROUPS.items():
            forbidden = set(features) & FORBIDDEN
            missing = [feature for feature in features if feature not in frame.columns]
            if forbidden or missing:
                raise ValueError(f"Invalid feature group {group_name}: forbidden={sorted(forbidden)}, missing={missing}")
            X_train = train_rows.loc[:, list(features)]
            X_validation = validation_rows.loc[:, list(features)]
            model, metrics = _fit_candidate(X_train, y_train, X_validation, y_validation, seed)
            candidate_models[group_name] = model
            candidate_results[group_name] = {
                "features": list(features),
                "feature_group_rationale": GROUP_RATIONALE[group_name],
                "timing": "unverified for every feature",
                "validation_metrics": metrics,
            }

        # AP is primary; F1 and fewer inputs are deterministic tie-breakers.
        winner_name = max(
            FEATURE_GROUPS,
            key=lambda name: (
                candidate_results[name]["validation_metrics"]["average_precision"],
                candidate_results[name]["validation_metrics"]["f1_Y"],
                -len(FEATURE_GROUPS[name]),
            ),
        )
        winner_model = candidate_models[winner_name]
        winner_features = list(FEATURE_GROUPS[winner_name])
        preprocessing = winner_model.named_steps["preprocess"]
        columns = classify_columns(train_rows.loc[:, winner_features])
        encoder = preprocessing.named_transformers_["categorical"].named_steps["encode"]
        categorical_options = {
            feature: [str(value) for value in values]
            for feature, values in zip(columns["categorical"], encoder.categories_)
        }
        integer_bounds = {}
        for feature in winner_features:
            if pd.api.types.is_integer_dtype(train_rows[feature].dtype):
                integer_bounds[feature] = [
                    int(train_rows[feature].min()), int(train_rows[feature].max())
                ]
        winning_metrics = candidate_results[winner_name]["validation_metrics"]
        specialist_bundle = {
            "pipeline": winner_model,
            "target": TARGET,
            "positive_label": POSITIVE_LABEL,
            "vehicle_category": category,
            "feature_group": winner_name,
            "input_features": winner_features,
            "feature_types": {feature: str(train_rows[feature].dtype) for feature in winner_features},
            "numeric_features": columns["numeric"],
            "ordinal_features": columns["ordinal"],
            "categorical_features": columns["categorical"],
            "categorical_options": categorical_options,
            "training_integer_bounds": integer_bounds,
            "decision_threshold_for_Y": winning_metrics["selected_threshold_for_Y"],
            "validation_metrics": winning_metrics,
            "source_sha256": source_hash,
            "split_manifest": manifest_path.as_posix(),
            "split_seed": seed,
            "training_rows": int(len(train_rows)),
            "validation_rows": int(len(validation_rows)),
            "test_used": False,
            "model_family": "RandomForestClassifier",
            "model_settings": {**RF_SETTINGS, "random_state": seed},
        }
        models_by_category[category] = specialist_bundle
        report_by_category[category] = {
            "training_counts": train_counts,
            "validation_counts": validation_counts,
            "candidate_groups": candidate_results,
            "selected_feature_group": winner_name,
            "selected_features": winner_features,
            "feature_retention_basis": [
                {
                    "column": feature,
                    "reason": (
                        f"Retained because it belongs to {winner_name}, the candidate group with "
                        f"the highest validation average precision ({winning_metrics['average_precision']:.6f}) "
                        "for this vehicle category. Grouped comparisons do not isolate individual feature effects."
                    ),
                }
                for feature in winner_features
            ],
            "selected_validation_metrics": winning_metrics,
            "timing_note": "Availability timing is unverified for every feature.",
            "test_used": False,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    report_path = output_dir / REPORT_FILENAME
    bundle = {
        "status": "exploratory classroom specialist models",
        "target": TARGET,
        "target_semantics_note": f"{TARGET} is an undocumented recorded label, not established confirmed fraud.",
        "source_sha256": source_hash,
        "split_manifest": manifest_path.as_posix(),
        "split_seed": seed,
        "partitions_used": {"train_rows": len(train_indices), "validation_rows": len(validation_indices), "test_used": False},
        "specialists": models_by_category,
        "test_evaluated": False,
    }
    report = {
        "status": "exploratory classroom specialist comparison",
        "target": TARGET,
        "target_semantics_note": f"The meaning of {TARGET} is undocumented. Models predict only that recorded label; they do not establish confirmed or real-world fraud detection.",
        "source_csv": csv_path.as_posix(),
        "source_sha256": source_hash,
        "split_manifest": manifest_path.as_posix(),
        "split_seed": seed,
        "partitions_used": {"train_rows": len(train_indices), "validation_rows": len(validation_indices), "test_used": False},
        "routing_field": {"column": ROUTING_FIELD, "use": "select specialist only; excluded from every pipeline input"},
        "candidate_groups": {
            name: {"features": list(features), "rationale": GROUP_RATIONALE[name]}
            for name, features in FEATURE_GROUPS.items()
        },
        "model_family_and_settings": {"family": "RandomForestClassifier", **RF_SETTINGS, "random_state": seed},
        "threshold_rule": "For each candidate, validation-only maximum F1 for the positive label; highest threshold breaks F1 ties.",
        "ranking_rule": "Within each vehicle category, highest validation average precision; validation F1 breaks a tie.",
        "preprocessing_and_fit_partition": "Each candidate preprocessing pipeline and classifier fit only on that category's training rows.",
        "feature_timing_note": "Timing and availability of every feature are unverified; candidate reduction is not a claim of screening-time availability.",
        "excluded_features": sorted(FORBIDDEN),
        "minimum_example_floor": {
            "training_per_class": MIN_TRAIN_PER_CLASS,
            "validation_per_class": MIN_VALIDATION_PER_CLASS,
            "note": "A modest run-safety floor only; passing it does not imply statistical reliability.",
        },
        "skipped_categories": skipped_categories,
        "specialists": report_by_category,
        "test_partition_used_for_fit_selection_threshold_or_metrics": False,
        "artifacts": {"model_bundle": model_path.as_posix(), "comparison_report": report_path.as_posix()},
    }
    joblib.dump(bundle, model_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = report_path.as_posix()
    report["model_path"] = model_path.as_posix()
    return report
