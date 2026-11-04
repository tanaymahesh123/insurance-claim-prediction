"""Reproducible, non-destructive structural and data-quality checks."""

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATE_COLUMNS, EXPECTED_COLUMNS, FEATURE_EXCLUSIONS

RARE_CATEGORY_MIN_COUNT = 5
IQR_MULTIPLIER = 1.5


def inspect_dataset(frame: pd.DataFrame, source: Path | str = "") -> dict[str, Any]:
    """Summarize schema and quality findings without changing the input frame.

    Numeric IQR findings and low-frequency categories are screening signals only;
    they are not treated as invalid values or evidence of an error.
    """
    actual_columns = list(frame.columns)
    missing_expected = [name for name in EXPECTED_COLUMNS if name not in actual_columns]
    unexpected_columns = [name for name in actual_columns if name not in EXPECTED_COLUMNS]

    column_details = []
    for name in actual_columns:
        column = frame[name]
        column_details.append(
            {
                "name": name,
                "dtype": str(column.dtype),
                "missing_count": int(column.isna().sum()),
                "missing_percent": float(column.isna().mean() * 100),
            }
        )

    numeric_screen = {}
    for name in frame.select_dtypes(include="number").columns:
        values = frame[name].dropna()
        if values.empty:
            numeric_screen[name] = {"iqr_outlier_count": 0, "note": "no non-missing values"}
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        spread = q3 - q1
        lower = q1 - IQR_MULTIPLIER * spread
        upper = q3 + IQR_MULTIPLIER * spread
        count = int(((values < lower) | (values > upper)).sum())
        numeric_screen[name] = {
            "iqr_outlier_count": count,
            "lower_fence": lower,
            "upper_fence": upper,
            "note": "statistical screening signal only; not an invalid-value judgment",
        }

    rare_categories = {}
    for name in frame.select_dtypes(exclude="number").columns:
        if name in FEATURE_EXCLUSIONS or name in DATE_COLUMNS:
            continue
        counts = Counter(frame[name].dropna().astype(str))
        rare_values = [(value, int(count)) for value, count in counts.items() if count < RARE_CATEGORY_MIN_COUNT]
        if rare_values:
            rare_categories[name] = {
                "rare_value_count": len(rare_values),
                "examples": [
                    {"value": value, "count": count}
                    for value, count in sorted(rare_values, key=lambda item: (item[1], item[0]))[:10]
                ],
            }

    date_checks = {}
    for name in DATE_COLUMNS:
        if name not in frame.columns:
            date_checks[name] = {"status": "column missing"}
            continue
        source_values = frame[name]
        parsed = pd.to_datetime(source_values, errors="coerce")
        invalid = source_values.notna() & parsed.isna()
        date_checks[name] = {
            "parsed_count": int(parsed.notna().sum()),
            "missing_source_count": int(source_values.isna().sum()),
            "parse_failure_count": int(invalid.sum()),
            "parse_failure_examples": source_values[invalid].astype(str).drop_duplicates().head(5).tolist(),
        }

    feature_exclusion_presence = [name for name in EXPECTED_COLUMNS if name in FEATURE_EXCLUSIONS and name in actual_columns]
    return {
        "source": Path(source).as_posix() if source else "",
        "row_count": int(len(frame)),
        "column_count": int(len(actual_columns)),
        "columns": column_details,
        "expected_columns_match": not missing_expected and not unexpected_columns,
        "missing_expected_columns": missing_expected,
        "unexpected_columns": unexpected_columns,
        "duplicate_row_count": int(frame.duplicated().sum()),
        "date_checks": date_checks,
        "numeric_iqr_screen": numeric_screen,
        "rare_categories_below_count": RARE_CATEGORY_MIN_COUNT,
        "rare_categories": rare_categories,
        "excluded_from_model_features": feature_exclusion_presence,
        "notes": [
            "No fields are assigned as a payout target by this check.",
            "IQR outliers and rare categories are review signals, not automatic errors.",
            "Schema validation uses the observed project CSV header; it does not establish field meaning or prediction timing.",
        ],
    }

