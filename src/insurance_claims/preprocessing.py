"""Leakage-aware preprocessing and reproducible train/validation/test splits."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import EXCLUDED_ALWAYS

# Each list runs from lowest to highest.
ORDINAL_CATEGORIES = {
    "VehiclePrice": [
        "less than 20000", "20000 to 29000", "30000 to 39000",
        "40000 to 59000", "60000 to 69000", "more than 69000",
    ],
    # The CSV splits the short range into "1 to 7" and "8 to 15".
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle": [
        "new", "2 years", "3 years", "4 years", "5 years",
        "6 years", "7 years", "more than 7",
    ],
    "AgeOfPolicyHolder": [
        "16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
        "36 to 40", "41 to 50", "51 to 65", "over 65",
    ],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": [
        "no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years",
    ],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
}


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with derived columns, skipping any whose source is absent."""
    result = df.copy()
    if "Fault" in result:
        result["fault_policy_holder"] = (result["Fault"] == "Policy Holder").astype("int64")
    if "PoliceReportFiled" in result:
        result["no_police_report"] = (result["PoliceReportFiled"] == "No").astype("int64")
    if "WitnessPresent" in result:
        result["no_witness"] = (result["WitnessPresent"] == "No").astype("int64")
    if "DriverRating" in result:
        result["high_driver_rating"] = (result["DriverRating"] >= 4).astype("int64")
    if "AgeOfVehicle" in result:
        order = {label: rank for rank, label in enumerate(ORDINAL_CATEGORIES["AgeOfVehicle"])}
        result["vehicle_age_num"] = result["AgeOfVehicle"].map(order)
    if "AddressChange_Claim" in result:
        result["address_changed_recently"] = (
            result["AddressChange_Claim"].isin(["under 6 months", "1 year"]).astype("int64")
        )
    return result


def prepare_features(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate a caller-specified label and drop identifier and demographic fields."""
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present in the dataset.")
    if target in EXCLUDED_ALWAYS:
        raise ValueError(f"{target!r} is an excluded column and cannot be the target.")
    remove = [column for column in EXCLUDED_ALWAYS if column in frame.columns]
    X = frame.drop(columns=[target, *remove]).copy()
    y = frame[target].copy()
    if X.empty:
        raise ValueError("No predictor columns remain after applying exclusions.")
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplit:
    """Create a stratified 70/15/15 split (adjusted for custom sizes)."""
    from sklearn.model_selection import train_test_split

    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must each be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_train_val,
    )
    return DatasetSplit(X_train, X_validation, X_test, y_train, y_validation, y_test)


def classify_columns(X: pd.DataFrame) -> dict[str, list[str]]:
    """Sort the columns present in X into numeric, ordinal, and categorical."""
    ordinal = [column for column in X.columns if column in ORDINAL_CATEGORIES]
    numeric = [
        column for column in X.select_dtypes(include=["number"]).columns
        if column not in ORDINAL_CATEGORIES
    ]
    categorical = [column for column in X.columns if column not in {*ordinal, *numeric}]
    return {"numeric": numeric, "ordinal": ordinal, "categorical": categorical}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a transformer for whichever numeric, ordinal, and nominal columns X has."""
    columns = classify_columns(X)

    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    # Unseen ordinal labels become missing, then take the training median.
    ordinal_pipeline = Pipeline(
        [
            (
                "encode",
                OrdinalEncoder(
                    categories=[ORDINAL_CATEGORIES[column] for column in columns["ordinal"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
            ),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, columns["numeric"]),
            ("ordinal", ordinal_pipeline, columns["ordinal"]),
            ("categorical", categorical_pipeline, columns["categorical"]),
        ],
        remainder="drop",
    )
