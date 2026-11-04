"""Rank existing and expanded models using only saved train/validation rows."""

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .comparison import build_xgb_pipeline
from .config import (
    DEFAULT_DATA_PATH,
    EXCLUDED_ALWAYS,
    FEATURES,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET,
)
from .data import load_dataset
from .preprocessing import build_preprocessor, classify_columns, engineer_features
from .training import _threshold_options

DEFAULT_MANIFEST = Path("data/processed/fraud_split_manifest.json")
DEFAULT_BASELINE_MODEL = Path("data/processed/fraud_logistic_pipeline.joblib")
DEFAULT_OUTPUT_DIR = Path("data/processed")
FORBIDDEN_FEATURES = set(EXCLUDED_ALWAYS) | {TARGET}
SHAP_REPORT_FILENAME = "fraud_shap_importance.json"
SHAP_TOP_N = 15
SHAP_DROP_FRACTION = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_scores(estimator, X: pd.DataFrame) -> np.ndarray:
    classes = list(estimator.classes_)
    if POSITIVE_LABEL not in classes:
        raise ValueError(f"Training labels do not include {POSITIVE_LABEL!r}.")
    return estimator.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]


def _select_threshold(y_validation: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_validation, scores, pos_label=POSITIVE_LABEL
    )
    if len(thresholds) == 0:
        raise ValueError("Validation data did not produce candidate thresholds.")
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    tied = np.flatnonzero(np.isclose(f1_values, f1_values.max()))
    chosen = int(tied[-1])  # same rule for every model: highest threshold among F1 ties
    return float(thresholds[chosen]), float(f1_values[chosen])


def _validation_metrics(
    y_validation: pd.Series, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = np.where(scores >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
    flagged = predictions == POSITIVE_LABEL
    return {
        "precision_Y": float(precision_score(y_validation, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall_Y": float(recall_score(y_validation, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_Y": float(f1_score(y_validation, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "average_precision": float(average_precision_score((y_validation == POSITIVE_LABEL).astype(int), scores)),
        "selected_threshold_for_Y": float(threshold),
        "validation_rows_predicted_Y": int(flagged.sum()),
        "validation_rows_predicted_Y_percent": float(flagged.mean() * 100),
        "validation_false_positive_count": int((flagged & (y_validation.to_numpy() == NEGATIVE_LABEL)).sum()),
    }


def _categoricals_to_int(X: pd.DataFrame) -> pd.DataFrame:
    """Cast encoded nominal columns to integers, which CatBoost requires for cat_features."""
    columns = [column for column in X.columns if column.startswith("categorical__")]
    return X.astype({column: "int64" for column in columns})


def _build_catboost_pipeline(X_train: pd.DataFrame, seed: int) -> Pipeline:
    columns = classify_columns(X_train)
    first_categorical = len(columns["numeric"]) + len(columns["ordinal"])
    cat_features_idx = list(range(first_categorical, first_categorical + len(columns["categorical"])))
    return Pipeline([
        ("preprocess", build_preprocessor(X_train).set_output(transform="pandas")),
        ("categoricals_to_int", FunctionTransformer(_categoricals_to_int)),
        ("classifier", CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            cat_features=cat_features_idx,
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        )),
    ])


def _shap_importance(pipeline: Pipeline, X_validation: pd.DataFrame) -> list[dict[str, float | str]]:
    """Return mean absolute SHAP per model input on validation rows, largest first."""
    preprocess = pipeline.named_steps["preprocess"]
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    shap_values = explainer.shap_values(preprocess.transform(X_validation))
    names = [str(name).split("__", 1)[1] for name in preprocess.get_feature_names_out()]
    importance = [
        {"feature": name, "mean_abs_shap": float(value)}
        for name, value in zip(names, np.abs(shap_values).mean(axis=0))
    ]
    return sorted(importance, key=lambda item: item["mean_abs_shap"], reverse=True)


def _make_pipeline(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> Pipeline:
    if model_name == "xgboost":
        return build_xgb_pipeline(X_train, y_train, seed)
    if model_name == "catboost":
        return _build_catboost_pipeline(X_train, seed)
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model {model_name!r}.")
    return Pipeline(
        [("preprocess", build_preprocessor(X_train)), ("classifier", classifier)]
    )


def rank_models_on_validation(
    csv_path: Path = DEFAULT_DATA_PATH,
    manifest_path: Path = DEFAULT_MANIFEST,
    baseline_model_path: Path = DEFAULT_BASELINE_MODEL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Fit candidate models on saved train rows and rank them on validation only."""
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

    saved_baseline = joblib.load(baseline_model_path)
    if saved_baseline.get("source_sha256") != source_hash:
        raise ValueError("Saved logistic model does not match the current CSV.")
    if saved_baseline.get("target") != TARGET or saved_baseline.get("input_features") != list(FEATURES):
        raise ValueError("Saved logistic model does not match the expected target and features.")

    review_items = manifest.get("availability_review_fields", [])
    review_features = {item["column"] for item in review_items}
    full_features = tuple(FEATURES)
    reduced_features = tuple(feature for feature in full_features if feature not in review_features)
    for name, features in (("full", full_features), ("reduced", reduced_features)):
        if set(features) & FORBIDDEN_FEATURES:
            raise ValueError(f"The {name} feature set includes a forbidden field.")

    frame = engineer_features(load_dataset(csv_path))
    required = set((*full_features, TARGET))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns are missing from the CSV: {missing}")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target {TARGET!r} contains missing values.")

    partitions = manifest["partitions"]
    train_indices = partitions["train"]["source_row_indices"]
    validation_indices = partitions["validation"]["source_row_indices"]
    if not train_indices or not validation_indices:
        raise ValueError("Saved train and validation partitions must be non-empty.")
    if len(set(train_indices)) != len(train_indices) or len(set(validation_indices)) != len(validation_indices):
        raise ValueError("Saved train or validation indices contain duplicates.")
    if set(train_indices) & set(validation_indices):
        raise ValueError("Saved train and validation indices overlap.")
    if min(train_indices + validation_indices) < 0 or max(train_indices + validation_indices) >= len(frame):
        raise ValueError("Saved train or validation index is outside the CSV row range.")

    y_train = frame.iloc[train_indices][TARGET]
    y_validation = frame.iloc[validation_indices][TARGET]
    full_set_name = f"full_{len(full_features)}"
    reduced_set_name = f"no_claim_timing_{len(reduced_features)}"
    feature_sets = {
        "full_logistic": ("logistic_regression", full_features, full_set_name),
        "full_random_forest": ("random_forest", full_features, full_set_name),
        "full_xgboost": ("xgboost", full_features, full_set_name),
        "full_catboost": ("catboost", full_features, full_set_name),
        "reduced_logistic": ("logistic_regression", reduced_features, reduced_set_name),
        "reduced_random_forest": ("random_forest", reduced_features, reduced_set_name),
        "reduced_xgboost": ("xgboost", reduced_features, reduced_set_name),
        "reduced_catboost": ("catboost", reduced_features, reduced_set_name),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_pipeline = saved_baseline["pipeline"]
    model_results: dict[str, Any] = {}
    model_artifacts: dict[str, Any] = {}
    fitted: dict[str, Any] = {}

    def evaluate(key: str, model_type: str, features: tuple[str, ...], set_name: str, estimator: Any = None) -> None:
        X_train = frame.iloc[train_indices].loc[:, list(features)]
        X_validation = frame.iloc[validation_indices].loc[:, list(features)]
        if key == "full_logistic":
            pipeline = baseline_pipeline
        else:
            pipeline = estimator if estimator is not None else _make_pipeline(
                model_type, X_train, y_train, int(manifest["seed"])
            )
            pipeline.fit(X_train, y_train)
        fitted[key] = pipeline

        scores = _positive_scores(pipeline, X_validation)
        threshold, validation_f1 = _select_threshold(y_validation, scores)
        if key == "full_logistic":
            stored_threshold = float(saved_baseline["decision_threshold_for_Y"])
            if not np.isclose(threshold, stored_threshold, rtol=0, atol=1e-12):
                raise ValueError("The saved baseline threshold differs from the common validation rule.")
            threshold = stored_threshold

        metrics = _validation_metrics(y_validation, scores, threshold)
        threshold_options = _threshold_options(y_validation, scores, threshold)
        numeric_bounds = {
            column: [float(X_train[column].min()), float(X_train[column].max())]
            for column in classify_columns(X_train)["numeric"]
        }
        model_results[key] = {
            "model": model_type,
            "feature_set": set_name,
            "features": list(features),
            "threshold_rule": "maximize validation F1 for the positive label; choose the highest threshold among ties",
            "validation_metrics": metrics,
            "threshold_options": threshold_options,
        }
        model_results[key]["review_feature_timing"] = [
            {"column": item["column"], "timing": "unverified"}
            for item in review_items if item["column"] in features
        ]

        artifact_name = f"fraud_ranked_{key}.joblib"
        artifact_path = output_dir / artifact_name
        if key == "full_logistic":
            bundle = {
                **saved_baseline,
                "comparison_name": key,
                "comparison_threshold_for_Y": threshold,
            }
        else:
            bundle = {
                "pipeline": pipeline,
                "target": TARGET,
                "positive_label": POSITIVE_LABEL,
                "input_features": list(features),
                "decision_threshold_for_Y": threshold,
                "source_sha256": source_hash,
                "split_manifest": manifest_path.as_posix(),
                "split_seed": int(manifest["seed"]),
                "model_name": model_type,
                "comparison_name": key,
            }
        bundle["threshold_options"] = threshold_options
        bundle["numeric_bounds"] = numeric_bounds
        model_artifacts[key] = artifact_path.as_posix()
        joblib.dump(bundle, artifact_path)

    def validation_ap(key: str) -> float:
        return model_results[key]["validation_metrics"]["average_precision"]

    for key, (model_type, features, set_name) in feature_sets.items():
        evaluate(key, model_type, features, set_name)

    # SHAP on the better XGBoost; the bottom share of all its inputs is removed from reduced_xgboost.
    best_xgb = max(("full_xgboost", "reduced_xgboost"), key=validation_ap)
    best_xgb_features = feature_sets[best_xgb][1]
    importance = _shap_importance(
        fitted[best_xgb], frame.iloc[validation_indices].loc[:, list(best_xgb_features)]
    )
    shap_top = importance[:SHAP_TOP_N]
    shap_path = output_dir / SHAP_REPORT_FILENAME
    shap_path.write_text(json.dumps(shap_top, indent=2) + "\n", encoding="utf-8")
    print(f"SHAP top {SHAP_TOP_N} ({best_xgb}, validation rows)")
    print(f"{'Feature':<28}{'Mean |SHAP|':>12}")
    for item in shap_top:
        print(f"{item['feature']:<28}{item['mean_abs_shap']:>12.4f}")

    bottom_count = int(len(importance) * SHAP_DROP_FRACTION)
    bottom_features = [item["feature"] for item in importance[-bottom_count:]] if bottom_count else []
    pruned_features = tuple(feature for feature in reduced_features if feature not in bottom_features)
    pruned_key = "reduced_xgboost_shap_pruned"
    evaluate(pruned_key, "xgboost", pruned_features, f"shap_pruned_{len(pruned_features)}")
    shap_pruning = {
        "explained_model": best_xgb,
        "bottom_fraction": SHAP_DROP_FRACTION,
        "bottom_features": bottom_features,
        "removed_from_reduced_xgboost": [f for f in reduced_features if f in bottom_features],
        "old_validation_average_precision": validation_ap("reduced_xgboost"),
        "new_validation_average_precision": validation_ap(pruned_key),
        "note": "Features were chosen using validation rows and rescored on those rows, so the new AP may be optimistic.",
    }
    print(
        f"reduced_xgboost AP {shap_pruning['old_validation_average_precision']:.4f} -> "
        f"{shap_pruning['new_validation_average_precision']:.4f} after removing "
        f"{shap_pruning['removed_from_reduced_xgboost']}"
    )

    best_rf = max(("full_random_forest", "reduced_random_forest"), key=validation_ap)
    stacking = StackingClassifier(
        estimators=[("xgb", clone(fitted[best_xgb])), ("rf", clone(fitted[best_rf]))],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
    )
    evaluate("stacking_xgb_rf", "stacking", full_features, f"{best_xgb}+{best_rf}", stacking)

    ranking = sorted(
        model_results,
        key=lambda name: (
            model_results[name]["validation_metrics"]["average_precision"],
            model_results[name]["validation_metrics"]["f1_Y"],
        ),
        reverse=True,
    )
    for rank, name in enumerate(ranking, start=1):
        model_results[name]["average_precision_rank"] = rank

    winner = ranking[0]
    validation_rows = len(validation_indices)
    report_path = output_dir / "fraud_validation_model_ranking.json"
    report: dict[str, Any] = {
        "status": "exploratory validation-only comparison",
        "target": TARGET,
        "target_semantics_note": f"The meaning of {TARGET} is undocumented. Results predict the dataset's recorded label only; they are not evidence of confirmed fraud or real-world fraud detection.",
        "feature_timing_note": "The timing of the claim-timing review fields is undocumented and unverified; the reduced feature set omits them.",
        "source_csv": csv_path.as_posix(),
        "source_sha256": source_hash,
        "split_manifest": manifest_path.as_posix(),
        "split_seed": int(manifest["seed"]),
        "partitions_used": {
            "train_rows": len(train_indices),
            "validation_rows": validation_rows,
            "test_used": False,
        },
        "threshold_selection_rule": "For each model, maximize validation F1 for the positive label; choose the highest threshold among ties.",
        "ranking_metric": "validation average precision (average precision summarizes ranking across thresholds)",
        "validation_rows_predicted_Y_percentage_denominator": validation_rows,
        "model_ranking": ranking,
        "models": model_results,
        "highest_validation_average_precision_model": winner,
        "winner_validation_metrics": model_results[winner]["validation_metrics"],
        "model_artifacts": model_artifacts,
        "shap_importance_report": shap_path.as_posix(),
        "shap_pruning": shap_pruning,
        "prohibited_features": sorted(FORBIDDEN_FEATURES),
        "demographic_features_excluded": True,
        "false_alarm_tradeoff_note": "Higher precision can result from flagging fewer rows; recall measures the share of positive validation rows that were found. Report flagged-row count and percentage with these metrics.",
        "validation_caveat": "Thresholds and models are compared on the same validation split, so ranked validation results may be optimistic.",
        "test_partition_used_for_fit_selection_threshold_or_metrics": False,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = report_path.as_posix()
    return report
