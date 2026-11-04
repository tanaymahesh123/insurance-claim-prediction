"""Classification metrics for severity and fraud-risk baselines."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def checked_metric(value: float) -> float:
    """Return a metric as a float, refusing any value outside the valid 0-1 range."""
    value = float(value)
    assert 0.0 <= value <= 1.01, f"Metric out of range: {value}"
    return value


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, Any]:
    """Return metrics useful for multiclass severity or binary fraud tasks."""
    labels = sorted(set(y_true) | set(y_pred), key=str)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [str(label) for label in labels],
    }

    if y_score is not None and len(labels) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
            positive_label = labels[-1]
            precision, recall, _, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[positive_label], zero_division=0
            )
            result["positive_class"] = str(positive_label)
            result["positive_class_precision"] = float(precision[0])
            result["positive_class_recall"] = float(recall[0])
        except (ValueError, IndexError):
            result["roc_auc"] = None
    return result
