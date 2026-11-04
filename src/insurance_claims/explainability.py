"""Inspection helpers for the fitted logistic regression baseline."""

import numpy as np
from sklearn.pipeline import Pipeline


def top_coefficients(model: Pipeline, limit: int = 15) -> list[dict[str, object]]:
    """Return features with the largest mean absolute logistic coefficients."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out()
    coefficients = np.atleast_2d(classifier.coef_)
    average_absolute = np.abs(coefficients).mean(axis=0)
    indices = np.argsort(average_absolute)[::-1][:limit]
    classes = [str(label) for label in classifier.classes_]

    return [
        {
            "feature": str(names[index]),
            "mean_absolute_coefficient": float(average_absolute[index]),
            "class_coefficients": {
                classes[row] if len(classes) == coefficients.shape[0] else str(row): float(
                    coefficients[row, index]
                )
                for row in range(coefficients.shape[0])
            },
        }
        for index in indices
    ]
