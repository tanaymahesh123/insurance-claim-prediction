"""Train and evaluate a narrow educational fraud-label baseline."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from .config import (
    DEFAULT_DATA_PATH,
    EXCLUDED_ALWAYS,
    FEATURES,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET,
)
from .data import load_dataset
from .preprocessing import build_preprocessor, engineer_features

RECALL_TARGET = 0.70
PRECISION_TARGET = 0.30
CV_FOLDS = 5
NESTED_OUTER_FOLDS = 5
NESTED_INNER_FOLDS = 3
FALSE_NEGATIVE_COST = 50
FALSE_POSITIVE_COST = 1
DEFAULT_MANIFEST = Path("data/processed/fraud_split_manifest.json")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(values: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in sorted(Counter(values).items(), key=lambda pair: str(pair[0]))
    }


def _positive_probability(estimator, X: pd.DataFrame) -> np.ndarray:
    classes = list(estimator.classes_)
    if POSITIVE_LABEL not in classes:
        raise ValueError(f"Training labels do not include {POSITIVE_LABEL!r}.")
    return estimator.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]


def _threshold_for_max_validation_f1(y_validation: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_validation, scores, pos_label=POSITIVE_LABEL
    )
    if len(thresholds) == 0:
        raise ValueError("Could not select a threshold from validation predictions.")
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best = np.flatnonzero(np.isclose(f1_values, f1_values.max()))
    # For tied validation F1, choose the higher threshold as the more conservative cutoff.
    chosen_index = int(best[-1])
    return float(thresholds[chosen_index]), float(f1_values[chosen_index])


def _operating_point(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    flagged = scores >= threshold
    actual = y_true.to_numpy() == POSITIVE_LABEL
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(actual, flagged, zero_division=0)),
        "recall": float(recall_score(actual, flagged, zero_division=0)),
        "f1": float(f1_score(actual, flagged, zero_division=0)),
        "rows_flagged": int(flagged.sum()),
        "false_positive_count": int((flagged & ~actual).sum()),
        "false_negative_count": int((~flagged & actual).sum()),
    }


def _cost_optimal_threshold(y_validation: pd.Series, scores: np.ndarray) -> float:
    """Minimize validation cost across observed scores; the highest threshold breaks ties."""
    actual = y_validation.to_numpy() == POSITIVE_LABEL
    candidates = np.unique(scores)
    costs = np.array([
        FALSE_NEGATIVE_COST * int((actual & (scores < candidate)).sum())
        + FALSE_POSITIVE_COST * int((~actual & (scores >= candidate)).sum())
        for candidate in candidates
    ])
    return float(candidates[np.flatnonzero(costs == costs.min())[-1]])


def _threshold_options(y_validation: pd.Series, scores: np.ndarray, max_f1_threshold: float) -> dict[str, Any]:
    """Find validation thresholds for each operating target; null when a target is unreachable."""
    precision, recall, thresholds = precision_recall_curve(
        y_validation, scores, pos_label=POSITIVE_LABEL
    )
    # thresholds ascend, and precision[i] and recall[i] describe thresholds[i].
    meets_recall = np.flatnonzero(recall[:-1] >= RECALL_TARGET)
    meets_precision = np.flatnonzero(precision[:-1] >= PRECISION_TARGET)
    chosen = {
        "threshold_max_f1": max_f1_threshold,
        "threshold_recall_70": float(thresholds[meets_recall[-1]]) if meets_recall.size else None,
        "threshold_precision_30": float(thresholds[meets_precision[0]]) if meets_precision.size else None,
        "threshold_cost_optimal": _cost_optimal_threshold(y_validation, scores),
    }
    notes = {}
    if chosen["threshold_recall_70"] is None:
        notes["threshold_recall_70"] = f"No validation threshold reaches recall >= {RECALL_TARGET}."
    if chosen["threshold_precision_30"] is None:
        notes["threshold_precision_30"] = f"No validation threshold reaches precision >= {PRECISION_TARGET}."
    return {
        **chosen,
        "partition": "validation only",
        "notes": notes,
        "validation_performance": {
            name: _operating_point(y_validation, scores, value)
            for name, value in chosen.items()
            if value is not None
        },
    }


def _cross_validation(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> dict[str, Any]:
    """Score a fresh copy of the pipeline on stratified folds of the training rows."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    results = cross_validate(
        pipeline, X_train, y_train, cv=folds, scoring={"average_precision": "average_precision", "f1": "f1"}
    )
    return {
        "folds": CV_FOLDS,
        "partition": "training rows only",
        "f1_threshold": "model default of 0.5",
        "mean_average_precision": float(results["test_average_precision"].mean()),
        "std_average_precision": float(results["test_average_precision"].std()),
        "mean_f1": float(results["test_f1"].mean()),
        "std_f1": float(results["test_f1"].std()),
    }


def _nested_cross_validation(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> dict[str, Any]:
    """Score each outer fold with a threshold chosen only from inner folds of its training rows."""
    outer_cv = StratifiedKFold(n_splits=NESTED_OUTER_FOLDS, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=NESTED_INNER_FOLDS, shuffle=True, random_state=seed)
    average_precisions, f1_values = [], []
    for outer_train, outer_validation in outer_cv.split(X_train, y_train):
        X_outer, y_outer = X_train.iloc[outer_train], y_train.iloc[outer_train]
        X_held_out, y_held_out = X_train.iloc[outer_validation], y_train.iloc[outer_validation]
        inner_scores = cross_val_predict(
            clone(pipeline), X_outer, y_outer, cv=inner_cv, method="predict_proba"
        )[:, 1]
        threshold, _ = _threshold_for_max_validation_f1(y_outer, inner_scores)
        fitted = clone(pipeline).fit(X_outer, y_outer)
        scores = _positive_probability(fitted, X_held_out)
        average_precisions.append(average_precision_score(y_held_out, scores))
        f1_values.append(f1_score(y_held_out, scores >= threshold, zero_division=0))
    return {
        "model": "LogisticRegression baseline",
        "outer_folds": NESTED_OUTER_FOLDS,
        "inner_folds": NESTED_INNER_FOLDS,
        "partition": "training rows only",
        "threshold_rule": "maximum F1 on inner-fold out-of-sample scores",
        "mean_average_precision": float(np.mean(average_precisions)),
        "std_average_precision": float(np.std(average_precisions)),
        "mean_f1_at_inner_threshold": float(np.mean(f1_values)),
        "std_f1_at_inner_threshold": float(np.std(f1_values)),
        "note": "Nested CV AP is a less optimistic estimate than the single validation split AP.",
        "threshold_note": "AP does not depend on the threshold, so only the F1 values reflect threshold selection.",
    }


def _metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = np.where(scores >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
    labels = [NEGATIVE_LABEL, POSITIVE_LABEL]
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=labels).ravel()
    return {
        "decision_threshold_for_Y": float(threshold),
        "precision_Y": float(precision_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall_Y": float(recall_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_Y": float(f1_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score((y_true == POSITIVE_LABEL).astype(int), scores)),
        "pr_auc_average_precision": float(average_precision_score((y_true == POSITIVE_LABEL).astype(int), scores)),
        "accuracy": float((predictions == np.asarray(y_true)).mean()),
        "confusion_matrix": {
            "labels": labels,
            "rows_actual_columns_predicted": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "predicted_counts": _counts(pd.Series(predictions)),
    }


def train_and_evaluate(
    csv_path: Path = DEFAULT_DATA_PATH,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Train on saved train indices, select threshold on validation, test once."""
    csv_path = Path(csv_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Saved split manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target") != TARGET:
        raise ValueError(f"Manifest target must be {TARGET!r}.")
    source_hash = _sha256(csv_path)
    if source_hash != manifest.get("source_sha256"):
        raise ValueError("CSV checksum differs from the source used to create the saved split.")

    invalid_features = sorted(set(FEATURES) & {*EXCLUDED_ALWAYS, TARGET})
    if invalid_features:
        raise ValueError(f"FEATURES contains excluded columns: {invalid_features}")

    frame = engineer_features(load_dataset(csv_path))
    missing = [column for column in (*FEATURES, TARGET) if column not in frame.columns]
    if missing:
        raise ValueError(f"Required columns are missing from the CSV: {missing}")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target {TARGET!r} contains missing values.")

    partitions = manifest["partitions"]
    train_indices = partitions["train"]["source_row_indices"]
    validation_indices = partitions["validation"]["source_row_indices"]
    test_indices = partitions["test"]["source_row_indices"]
    all_indices = train_indices + validation_indices + test_indices
    if len(all_indices) != len(frame) or len(set(all_indices)) != len(frame):
        raise ValueError("Saved split indices do not form a complete, non-overlapping partition of this CSV.")
    if min(all_indices) < 0 or max(all_indices) >= len(frame):
        raise ValueError("Saved split indices are outside the CSV row range.")

    X_train = frame.iloc[train_indices].loc[:, FEATURES]
    y_train = frame.iloc[train_indices][TARGET]
    X_validation = frame.iloc[validation_indices].loc[:, FEATURES]
    y_validation = frame.iloc[validation_indices][TARGET]

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(manifest["seed"]))),
        ]
    )
    pipeline.fit(X_train, y_train)

    validation_scores = _positive_probability(pipeline, X_validation)
    threshold, validation_f1 = _threshold_for_max_validation_f1(y_validation, validation_scores)
    threshold_options = _threshold_options(y_validation, validation_scores, threshold)
    cross_validation = _cross_validation(pipeline, X_train, y_train, int(manifest["seed"]))
    nested_cv = _nested_cross_validation(pipeline, X_train, y_train, int(manifest["seed"]))

    # Test labels/features are first accessed after the threshold has been fixed.
    X_test = frame.iloc[test_indices].loc[:, FEATURES]
    y_test = frame.iloc[test_indices][TARGET]
    test_scores = _positive_probability(pipeline, X_test)
    logistic_test_metrics = _metrics(y_test, test_scores, threshold)

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)
    majority_scores = _positive_probability(majority, X_test)
    majority_label = str(majority.classes_[int(np.argmax(majority.class_prior_))])
    majority_threshold = 0.0 if majority_label == POSITIVE_LABEL else 1.0
    majority_test_metrics = _metrics(y_test, majority_scores, majority_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "fraud_logistic_pipeline.joblib"
    report_path = output_dir / "fraud_baseline_evaluation.json"
    report: dict[str, Any] = {
        "task": f"classification of the dataset's recorded {TARGET} label",
        "target": TARGET,
        "target_values": manifest["target_values_observed"],
        "target_counts": manifest["target_counts"],
        "target_semantics_note": "Label meaning is undocumented; results are predictions of the recorded label and must not be described as confirmed fraud or real-world fraud risk.",
        "source_csv": csv_path.as_posix(),
        "source_sha256": source_hash,
        "split_manifest": manifest_path.as_posix(),
        "split_seed": manifest["seed"],
        "split_proportions": manifest["split_proportions"],
        "split_row_counts": {name: int(len(partitions[name]["source_row_indices"])) for name in ("train", "validation", "test")},
        "input_features": list(FEATURES),
        "excluded_features": list(EXCLUDED_ALWAYS),
        "feature_availability_note": "Availability and timing of these inputs at the intended prediction moment are unverified; see availability_review_fields in the split manifest.",
        "preprocessing": {
            "fit_partition": "train",
            "numeric_imputation": "training median",
            "scaling": "training mean and standard deviation",
            "ordinal": "mapped to integers in the configured order, then treated as numeric",
            "categorical": "ordinal encoded; unknown categories become -1",
            "validation_or_test_used_in_fit": False,
        },
        "baseline_model": "LogisticRegression with class_weight='balanced'",
        "baseline_comparison": "DummyClassifier(strategy='most_frequent') fitted on training labels",
        "threshold_selection": {
            "partition": "validation only",
            "criterion": "maximum F1 for the positive label",
            "tie_break": "highest threshold among tied maximum validation F1 values",
            "selected_threshold": threshold,
            "validation_f1_Y": validation_f1,
        },
        "threshold_options": threshold_options,
        "cross_validation": cross_validation,
        "nested_cv": nested_cv,
        "test_metrics": {
            "logistic_regression": logistic_test_metrics,
            "majority_class": majority_test_metrics,
        },
        "metric_notes": {
            "positive_label": POSITIVE_LABEL,
            "pr_auc_average_precision": "Average precision, a step-wise precision-recall summary computed from scores.",
            "confusion_matrix": "Rows are actual 0, 1; columns are predicted 0, 1.",
            "accuracy": "Included for context; it can look high when the majority 0 class dominates while 1 cases are missed.",
        },
        "model_trained": True,
        "test_used_for_fitting_or_threshold_selection": False,
        "saved_pipeline": model_path.as_posix(),
    }
    joblib.dump(
        {
            "pipeline": pipeline,
            "target": TARGET,
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "input_features": list(FEATURES),
            "decision_threshold_for_Y": threshold,
            "threshold_options": threshold_options,
            "source_sha256": source_hash,
            "split_manifest": manifest_path.as_posix(),
            "split_seed": manifest["seed"],
        },
        model_path,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["saved_report"] = report_path.as_posix()
    return report
