"""Run bounded, exploratory feature and threshold experiments on specialists.

All fitting uses saved training rows; comparisons use saved validation rows.
The test partition is deliberately excluded from all calculations.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from .config import DEFAULT_DATA_PATH, EXCLUDED_ALWAYS, NEGATIVE_LABEL, POSITIVE_LABEL, TARGET
from .data import load_dataset
from .metrics import checked_metric
from .preprocessing import build_preprocessor, engineer_features

DEFAULT_MANIFEST = Path("data/processed/fraud_split_manifest.json")
DEFAULT_SPECIALISTS = Path("data/processed/fraud_vehicle_specialists_v1.joblib")
DEFAULT_OUTPUT_DIR = Path("data/processed")
MODEL_FILENAME = "fraud_vehicle_specialist_error_experiments_v1.joblib"
REPORT_FILENAME = "fraud_vehicle_specialist_error_experiments_v1.json"
ROUTING_FIELD = "VehicleCategory"
FORBIDDEN = set(EXCLUDED_ALWAYS) | {TARGET, ROUTING_FIELD}
LOWERED_FIXED_THRESHOLDS = (0.40, 0.25)

# Each ablation removes a raw field together with the indicator derived from it.
ABLATIONS = {
    "remove_witness_signals": ("WitnessPresent", "no_witness"),
    "remove_fault_signals": ("Fault", "fault_policy_holder"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_scores(pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(pipeline.named_steps["classifier"].classes_)
    return pipeline.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]


def _threshold_metrics(y: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    actual_y = (y.to_numpy() == POSITIVE_LABEL)
    predicted_y = scores >= threshold
    return {
        "threshold": float(threshold),
        "precision_Y": checked_metric(precision_score(actual_y, predicted_y, zero_division=0)),
        "recall_Y": checked_metric(recall_score(actual_y, predicted_y, zero_division=0)),
        "f1_Y": checked_metric(f1_score(actual_y, predicted_y, zero_division=0)),
        "validation_rows_predicted_Y": int(predicted_y.sum()),
        "validation_rows_predicted_Y_percent": float(predicted_y.mean() * 100),
        "false_positive_count": int((predicted_y & ~actual_y).sum()),
        "false_negative_count": int((~predicted_y & actual_y).sum()),
    }


def _fit_random_forest(X: pd.DataFrame, y: pd.Series, model_settings: dict[str, Any]) -> Pipeline:
    settings = dict(model_settings)
    settings.setdefault("n_jobs", -1)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("classifier", RandomForestClassifier(**settings)),
    ])
    pipeline.fit(X, y)
    return pipeline


def run_specialist_error_experiments(
    csv_path: Path = DEFAULT_DATA_PATH,
    manifest_path: Path = DEFAULT_MANIFEST,
    specialists_path: Path = DEFAULT_SPECIALISTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Fit only the preregistered variants and score them on validation rows."""
    csv_path, manifest_path = Path(csv_path), Path(manifest_path)
    specialists_path, output_dir = Path(specialists_path), Path(output_dir)
    for path in (csv_path, manifest_path, specialists_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input not found: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved = joblib.load(specialists_path)
    source_hash = _sha256(csv_path)
    if manifest.get("target") != TARGET or saved.get("target") != TARGET:
        raise ValueError("Experiment inputs do not use the expected recorded target.")
    if manifest.get("source_sha256") != source_hash or saved.get("source_sha256") != source_hash:
        raise ValueError("CSV, split manifest, and saved specialists must reference the same source.")

    frame = engineer_features(load_dataset(csv_path))
    parts = manifest["partitions"]
    train_indices = [int(i) for i in parts["train"]["source_row_indices"]]
    validation_indices = [int(i) for i in parts["validation"]["source_row_indices"]]
    test_indices = set(int(i) for i in parts["test"]["source_row_indices"])
    if set(train_indices) & set(validation_indices) or (set(train_indices) | set(validation_indices)) & test_indices:
        raise ValueError("Saved train, validation, and test assignments must be disjoint.")
    if not train_indices or not validation_indices or max(train_indices + validation_indices) >= len(frame):
        raise ValueError("Saved training or validation assignments are empty or outside the source data.")
    train = frame.iloc[train_indices]
    validation = frame.iloc[validation_indices]

    results: dict[str, Any] = {}
    new_pipelines: dict[str, Any] = {}
    for category, current in saved["specialists"].items():
        train_rows = train[train[ROUTING_FIELD].astype(str) == category]
        validation_rows = validation[validation[ROUTING_FIELD].astype(str) == category]
        y_train = train_rows[TARGET]
        y_validation = validation_rows[TARGET]
        base_features = list(current["input_features"])
        forbidden = set(base_features) & FORBIDDEN
        if forbidden:
            raise ValueError(f"Saved specialist for {category} uses forbidden inputs: {sorted(forbidden)}")
        model_variants: dict[str, tuple[Any, list[str], str]] = {
            "current_saved_model": (
                current["pipeline"], base_features,
                "Unmodified pipeline loaded from the current saved specialist artifact.",
            )
        }

        settings = dict(current["model_settings"])
        for variant_name, removed in ABLATIONS.items():
            reduced_features = [feature for feature in base_features if feature not in removed]
            if len(reduced_features) == len(base_features):
                continue
            pipeline = _fit_random_forest(train_rows.loc[:, reduced_features], y_train, settings)
            model_variants[variant_name] = (
                pipeline,
                reduced_features,
                f"Refits the same Random Forest after removing {', '.join(removed)}.",
            )
            new_pipelines[f"{category}::{variant_name}"] = {
                "pipeline": pipeline,
                "input_features": reduced_features,
                "removed_features": list(removed),
                "model_settings": settings,
                "training_rows": int(len(train_rows)),
            }

        thresholds = [
            {"name": "saved_current_threshold", "value": float(current["decision_threshold_for_Y"]),
             "purpose": "unchanged reference cutoff saved with the current model"},
            *[
                {"name": f"lower_fixed_{value:.2f}".replace(".", "_"), "value": value,
                 "purpose": "predeclared lower cutoff intended to test a higher-recall trade-off"}
                for value in LOWERED_FIXED_THRESHOLDS
            ],
        ]
        variant_results: dict[str, Any] = {}
        for variant_name, (pipeline, features, rationale) in model_variants.items():
            scores = _positive_scores(pipeline, validation_rows.loc[:, features])
            variant_results[variant_name] = {
                "input_features": features,
                "rationale": rationale,
                "validation_average_precision": checked_metric(
                    average_precision_score((y_validation == POSITIVE_LABEL).astype(int), scores)
                ),
                "threshold_results": [
                    {"threshold_name": item["name"], **_threshold_metrics(y_validation, scores, item["value"])}
                    for item in thresholds
                ],
            }
        results[category] = {
            "training_rows": int(len(train_rows)),
            "validation_rows": int(len(validation_rows)),
            "training_target_counts": {str(label): int((y_train == label).sum()) for label in (NEGATIVE_LABEL, POSITIVE_LABEL)},
            "validation_target_counts": {str(label): int((y_validation == label).sum()) for label in (NEGATIVE_LABEL, POSITIVE_LABEL)},
            "thresholds_compared": thresholds,
            "variants": variant_results,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    report_path = output_dir / REPORT_FILENAME
    experiment_bundle = {
        "status": "exploratory validation error-pattern experiment",
        "target": TARGET,
        "source_sha256": source_hash,
        "split_manifest": manifest_path.as_posix(),
        "reference_specialist_artifact": specialists_path.as_posix(),
        "thresholds_selected_on_test": False,
        "test_used_for_fit_selection_or_metrics": False,
        "new_pipelines": new_pipelines,
    }
    report = {
        "status": "exploratory validation error-pattern experiment; potentially overfit",
        "target": TARGET,
        "target_semantics_note": f"{TARGET} is an undocumented recorded label; a positive label does not mean confirmed fraud.",
        "feature_timing_note": "Availability timing for these features is unverified; this is not evidence that a feature is known at screening time.",
        "source_csv": csv_path.as_posix(),
        "source_sha256": source_hash,
        "split_manifest": manifest_path.as_posix(),
        "reference_specialist_artifact": specialists_path.as_posix(),
        "partitions_used": {
            "training_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_used_for_fit_model_selection_threshold_or_metrics": False,
        },
        "predeclared_experiments": {
            "feature_ablations": {name: list(removed) for name, removed in ABLATIONS.items()},
            "threshold_comparison": "Apply each current saved threshold plus fixed lower cutoffs 0.40 and 0.25; no threshold is optimized or selected in this experiment.",
            "model_family": "RandomForestClassifier with the current specialist's saved settings.",
        },
        "excluded_features": sorted(FORBIDDEN),
        "validation_caveat": "Earlier validation results informed existing feature and threshold choices. These additional validation comparisons are exploratory and potentially overfit.",
        "specialists": results,
        "artifacts": {"models": model_path.as_posix(), "report": report_path.as_posix()},
    }
    joblib.dump(experiment_bundle, model_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["model_path"] = model_path.as_posix()
    report["report_path"] = report_path.as_posix()
    return report
