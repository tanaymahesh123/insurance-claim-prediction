"""Exploratory validation-only comparison of logistic, Random Forest, and XGBoost models."""

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
from xgboost import XGBClassifier

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

DEFAULT_MANIFEST = Path("data/processed/fraud_split_manifest.json")
DEFAULT_BASELINE_MODEL = Path("data/processed/fraud_logistic_pipeline.joblib")
DEFAULT_OUTPUT_DIR = Path("data/processed")
PROHIBITED_FIELDS = set(EXCLUDED_ALWAYS) | {TARGET}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_scores(estimator, X: pd.DataFrame) -> np.ndarray:
    classes = list(estimator.classes_)
    if POSITIVE_LABEL not in classes:
        raise ValueError(f"Training labels do not contain {POSITIVE_LABEL!r}.")
    return estimator.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]


def _f1_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores, pos_label=POSITIVE_LABEL
    )
    if len(thresholds) == 0:
        raise ValueError("Validation data did not produce any candidate thresholds.")
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    tied = np.flatnonzero(np.isclose(f1_values, f1_values.max()))
    index = int(tied[-1])  # highest threshold among F1 ties
    return float(thresholds[index]), float(f1_values[index])


def _validation_metrics(
    y_true: pd.Series, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = np.where(scores >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
    return {
        "precision_Y": float(precision_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall_Y": float(recall_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_Y": float(f1_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "average_precision": float(average_precision_score((y_true == POSITIVE_LABEL).astype(int), scores)),
        "selected_threshold_for_Y": float(threshold),
        "validation_predicted_Y_count": int((predictions == POSITIVE_LABEL).sum()),
        "validation_false_positive_count": int(((predictions == POSITIVE_LABEL) & (y_true.to_numpy() == NEGATIVE_LABEL)).sum()),
    }


def build_xgb_pipeline(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
    """Build the XGBoost pipeline, weighting positives by the training class ratio."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    return Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            eval_metric="aucpr",
            random_state=seed,
            verbosity=0,
        )),
    ])


def _build_random_forest_pipeline(X_train: pd.DataFrame, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(X_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def compare_on_validation(
    csv_path: Path = DEFAULT_DATA_PATH,
    manifest_path: Path = DEFAULT_MANIFEST,
    baseline_model_path: Path = DEFAULT_BASELINE_MODEL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Compare models using only the saved training and validation rows."""
    csv_path = Path(csv_path)
    manifest_path = Path(manifest_path)
    baseline_model_path = Path(baseline_model_path)
    output_dir = Path(output_dir)

    for path in (csv_path, manifest_path, baseline_model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison input not found: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target") != TARGET:
        raise ValueError(f"Saved split manifest target must be {TARGET!r}.")
    source_hash = _sha256(csv_path)
    if source_hash != manifest.get("source_sha256"):
        raise ValueError("CSV checksum differs from the source used to create the saved split.")

    baseline_bundle = joblib.load(baseline_model_path)
    if baseline_bundle.get("source_sha256") != source_hash:
        raise ValueError("Saved logistic model does not match the current CSV.")
    if baseline_bundle.get("target") != TARGET:
        raise ValueError(f"Saved logistic model target must be {TARGET!r}.")
    if baseline_bundle.get("input_features") != list(FEATURES):
        raise ValueError("Saved logistic model does not use the configured features.")
    if set(FEATURES) & PROHIBITED_FIELDS:
        raise ValueError("FEATURES includes a prohibited or demographic field.")

    frame = engineer_features(load_dataset(csv_path))
    missing = [column for column in (*FEATURES, TARGET) if column not in frame.columns]
    if missing:
        raise ValueError(f"Required comparison columns are missing: {missing}")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target {TARGET!r} contains missing values.")

    partitions = manifest["partitions"]
    train_indices = partitions["train"]["source_row_indices"]
    validation_indices = partitions["validation"]["source_row_indices"]
    if len(set(train_indices)) != len(train_indices) or len(set(validation_indices)) != len(validation_indices):
        raise ValueError("Saved training or validation indices contain duplicates.")
    if set(train_indices) & set(validation_indices):
        raise ValueError("Saved training and validation indices overlap.")
    if not train_indices or not validation_indices:
        raise ValueError("Saved training and validation partitions must be non-empty.")
    if min(train_indices + validation_indices) < 0 or max(train_indices + validation_indices) >= len(frame):
        raise ValueError("Saved training or validation index is outside the CSV row range.")

    X_train = frame.iloc[train_indices].loc[:, FEATURES]
    X_validation = frame.iloc[validation_indices].loc[:, FEATURES]
    y_train = frame.iloc[train_indices][TARGET]
    y_validation = frame.iloc[validation_indices][TARGET]

    # The logistic model keeps the threshold saved by training, which came from the
    # same validation rule the tree models use below.
    seed = int(manifest["seed"])
    baseline_threshold = float(baseline_bundle["decision_threshold_for_Y"])
    baseline_metrics = _validation_metrics(
        y_validation, _positive_scores(baseline_bundle["pipeline"], X_validation), baseline_threshold
    )
    candidates = {
        "random_forest": _build_random_forest_pipeline(X_train, seed),
        "xgboost": build_xgb_pipeline(X_train, y_train, seed),
    }
    models: dict[str, Any] = {
        "logistic_regression": {
            "model": "saved LogisticRegression pipeline with class_weight='balanced'",
            "features": list(FEATURES),
            "threshold_selection": {"partition": "validation only", "selected_threshold_for_Y": baseline_threshold},
            "validation_metrics": baseline_metrics,
        }
    }
    fitted: dict[str, Pipeline] = {"logistic_regression": baseline_bundle["pipeline"]}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        scores = _positive_scores(pipeline, X_validation)
        threshold, maximum_f1 = _f1_threshold(y_validation, scores)
        fitted[name] = pipeline
        models[name] = {
            "model": type(pipeline.named_steps["classifier"]).__name__,
            "features": list(FEATURES),
            "threshold_selection": {
                "partition": "validation only",
                "criterion": "maximum F1 for the positive label",
                "tie_break": "highest threshold among tied maximum validation F1 values",
                "selected_threshold_for_Y": threshold,
                "maximum_validation_f1_Y": maximum_f1,
            },
            "validation_metrics": _validation_metrics(y_validation, scores, threshold),
        }

    # Average precision ranks the models; validation F1 breaks a tie.
    ranking = sorted(
        models,
        key=lambda name: (
            models[name]["validation_metrics"]["average_precision"],
            models[name]["validation_metrics"]["f1_Y"],
        ),
        reverse=True,
    )
    for rank, name in enumerate(ranking, start=1):
        models[name]["average_precision_rank"] = rank
    selected_name = ranking[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        name: output_dir / f"fraud_exploratory_{name}.joblib" for name in ranking
    }
    report_path = output_dir / "fraud_exploratory_feature_comparison.json"
    joblib.dump(baseline_bundle, artifact_paths["logistic_regression"])
    for name in candidates:
        joblib.dump(
            {
                "pipeline": fitted[name],
                "target": TARGET,
                "positive_label": POSITIVE_LABEL,
                "input_features": list(FEATURES),
                "decision_threshold_for_Y": models[name]["threshold_selection"]["selected_threshold_for_Y"],
                "source_sha256": source_hash,
                "split_manifest": manifest_path.as_posix(),
                "split_seed": seed,
                "model_name": models[name]["model"],
            },
            artifact_paths[name],
        )

    comparison: dict[str, Any] = {
        "status": "exploratory classroom comparison",
        "target": TARGET,
        "target_semantics_note": f"The meaning of {TARGET} is undocumented. These models classify the dataset's recorded label only; they do not establish confirmed fraud or real-world fraud detection.",
        "source_csv": csv_path.as_posix(),
        "source_sha256": source_hash,
        "split_manifest": manifest_path.as_posix(),
        "split_seed": seed,
        "partitions_used": {
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_used": False,
        },
        "model_ranking": ranking,
        "models": {name: models[name] for name in ranking},
        "comparison_policy": {
            "model_selection_measure": "validation average precision; validation F1 breaks a tie",
            "validation_metrics_are_selection_set_results": True,
            "validation_caveat": "Thresholds and models are selected on this same validation split, so threshold-based validation metrics are optimistic.",
            "preprocessing_fit_partition": "train only",
            "model_fit_partition": "train only",
            "test_partition_used_for_fitting_selection_or_metrics": False,
            "feature_timing_note": "Availability timing of these inputs is unverified; see availability_review_fields in the split manifest.",
            "prohibited_fields_excluded": sorted(PROHIBITED_FIELDS),
            "false_alarm_note": "At a chosen threshold, more positive predictions can improve recall while increasing false positives. Review validation precision and false-positive counts before drawing conclusions.",
        },
        "selected_model_by_validation_only": selected_name,
        "artifacts": {
            **{name: path.as_posix() for name, path in artifact_paths.items()},
            "selected_model": artifact_paths[selected_name].as_posix(),
            "comparison_report": report_path.as_posix(),
        },
        "test_evaluated": False,
    }
    report_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return comparison
