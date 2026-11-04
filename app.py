"""Local demo: score a claim with the top validation-ranked model and the Sedan specialist."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insurance_claims.preprocessing import ORDINAL_CATEGORIES, engineer_features  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
RANKING_PATH = PROCESSED / "fraud_validation_model_ranking.json"
SPECIALIST_MODEL_PATH = PROCESSED / "fraud_vehicle_specialists_v1.joblib"
SPECIALIST_REPORT_PATH = PROCESSED / "fraud_vehicle_specialists_v1.json"

GLOBAL_MODEL = "reduced_catboost"
ROUTING_FIELD = "VehicleCategory"
VEHICLE_CATEGORIES = ("Sedan", "Sport", "Utility")
POSITIVE_LABEL = 1
# Computed by engineer_features from the fields the user enters.
DERIVED_FIELDS = {
    "fault_policy_holder",
    "no_police_report",
    "no_witness",
    "high_driver_rating",
    "address_changed_recently",
}
# "0" appears in these columns in the raw data but is an artifact, not a real input.
ARTIFACT_OPTIONS = {"MonthClaimed": "0", "DayOfWeekClaimed": "0"}
THRESHOLD_LABELS = {
    "threshold_max_f1": "max_f1",
    "threshold_recall_70": "recall_70",
    "threshold_precision_30": "precision_30",
    "threshold_cost_optimal": "Cost-optimal (50:1 FN:FP ratio)",
}
GLOBAL_MODEL_NOTE = "Sport and Utility use the global model (too few training examples for a specialist)."


@st.cache_resource
def load_artifacts() -> tuple[dict, dict, dict, dict]:
    """Load the global model and the specialists from trusted local artifacts."""
    for path in (RANKING_PATH, SPECIALIST_MODEL_PATH, SPECIALIST_REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.name}. Run the prepare, train, compare, and specialist commands in the README."
            )
    ranking = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
    global_bundle = joblib.load(PROJECT_ROOT / ranking["model_artifacts"][GLOBAL_MODEL])
    specialists = joblib.load(SPECIALIST_MODEL_PATH)
    specialist_report = json.loads(SPECIALIST_REPORT_PATH.read_text(encoding="utf-8"))
    hashes = {
        ranking["source_sha256"],
        global_bundle["source_sha256"],
        specialists["source_sha256"],
        specialist_report["source_sha256"],
    }
    if len(hashes) != 1:
        raise ValueError("Saved artifacts were built from different source data. Re-run the pipeline.")
    return ranking, global_bundle, specialists, specialist_report


def field_options(pipeline) -> dict[str, list[str]]:
    """Read dropdown choices for ordinal and nominal inputs from a fitted pipeline."""
    preprocess = pipeline.named_steps["preprocess"]
    options: dict[str, list[str]] = {}
    for name, _, columns in preprocess.transformers_:
        if name == "ordinal":
            options.update({column: ORDINAL_CATEGORIES[column] for column in columns})
        elif name == "categorical":
            encoder = preprocess.named_transformers_["categorical"].named_steps["encode"]
            options.update({
                column: [str(value) for value in values if str(value) != ARTIFACT_OPTIONS.get(column)]
                for column, values in zip(columns, encoder.categories_)
            })
    return options


def score(bundle: dict, values: dict[str, object]) -> float:
    """Return the model's positive-class output for one entered claim."""
    row = engineer_features(pd.DataFrame([values]))
    pipeline = bundle["pipeline"]
    position = list(pipeline.classes_).index(POSITIVE_LABEL)
    return float(pipeline.predict_proba(row.loc[:, bundle["input_features"]])[0, position])


def collect_inputs(models: list[dict], category: str) -> tuple[bool, dict[str, object]]:
    """Render one required input per raw field that the given models need."""
    fields: list[str] = []
    for bundle in models:
        fields += [
            field for field in bundle["input_features"]
            if field not in fields and field not in DERIVED_FIELDS and field != ROUTING_FIELD
        ]
    options: dict[str, list[str]] = {}
    bounds: dict[str, list[float]] = {}
    for bundle in reversed(models):
        options.update(field_options(bundle["pipeline"]))
        bounds.update(bundle.get("training_integer_bounds", {}))
        bounds.update(bundle.get("numeric_bounds", {}))

    values: dict[str, object] = {}
    with st.form(f"claim_{category}"):
        left, right = st.columns(2)
        for index, field in enumerate(fields):
            with left if index % 2 == 0 else right:
                if field in options:
                    values[field] = st.selectbox(
                        field, options[field], index=None, placeholder="Select an option", key=f"{category}_{field}"
                    )
                else:
                    lower, upper = bounds[field]
                    values[field] = st.number_input(
                        field, min_value=int(lower), max_value=int(upper), value=None, step=1,
                        key=f"{category}_{field}", help=f"Training range: {int(lower)} to {int(upper)}",
                    )
        submitted = st.form_submit_button("Get prediction", type="primary")
    return submitted, values


def show_threshold_results(probability: float, options: dict) -> None:
    """Let the user switch operating thresholds and see how the decision changes."""
    choice = st.radio(
        "Decision threshold", list(THRESHOLD_LABELS), format_func=THRESHOLD_LABELS.get, horizontal=True
    )
    rows = []
    for key, label in THRESHOLD_LABELS.items():
        threshold = options[key]
        performance = options["validation_performance"].get(key)
        rows.append({
            "Option": label,
            "Threshold": threshold,
            "This claim": "n/a" if threshold is None else ("Flagged" if probability >= threshold else "Not flagged"),
            "Val precision": performance["precision"] if performance else None,
            "Val recall": performance["recall"] if performance else None,
            "Val rows flagged": performance["rows_flagged"] if performance else None,
        })
    threshold = options[choice]
    if threshold is None:
        st.warning(options["notes"].get(choice, "This threshold is not available for this model."))
    elif probability >= threshold:
        st.error(f"Flagged for review at {THRESHOLD_LABELS[choice]} (threshold {threshold:.4f}).")
    else:
        st.success(f"Not flagged at {THRESHOLD_LABELS[choice]} (threshold {threshold:.4f}).")
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_specialist(probability: float, specialist: dict, details: dict) -> None:
    """Show the Sedan specialist beside the global result, at its own saved threshold."""
    st.subheader(f"{specialist['vehicle_category']} specialist")
    threshold = float(specialist["decision_threshold_for_Y"])
    metrics = details["selected_validation_metrics"]
    first, second, third = st.columns(3)
    first.metric("Predicted fraud probability", f"{probability:.1%}")
    second.metric("Score (0-100)", f"{probability * 100:.0f}")
    third.metric("Decision", "Flagged" if probability >= threshold else "Not flagged")
    st.caption(
        f"Random Forest, feature group `{specialist['feature_group']}`, fixed max-F1 threshold {threshold:.4f}. "
        f"Validation AP {metrics['average_precision']:.4f}, precision {metrics['precision_Y']:.1%}, "
        f"recall {metrics['recall_Y']:.1%}. The threshold options above apply to the global model only."
    )
    if specialist["feature_group"] == "full":
        st.caption("This specialist uses claim-timing fields whose availability at prediction time is unverified.")


def main() -> None:
    st.set_page_config(page_title="Insurance Claim Fraud App", page_icon="📊", layout="centered")
    st.title("Insurance Claim Fraud App")
    try:
        ranking, global_bundle, specialists, specialist_report = load_artifacts()
    except (FileNotFoundError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        st.stop()

    global_metrics = ranking["models"][GLOBAL_MODEL]["validation_metrics"]
    st.write(f"Global model: **{GLOBAL_MODEL}** (validation AP {global_metrics['average_precision']:.4f})")

    category = st.selectbox(ROUTING_FIELD, VEHICLE_CATEGORIES)
    specialist = specialists["specialists"].get(category)
    models = [global_bundle] + ([specialist] if specialist else [])
    submitted, values = collect_inputs(models, category)

    if submitted:
        missing = [field for field, value in values.items() if value is None]
        if missing:
            st.session_state.pop("result", None)
            st.error("Please fill in: " + ", ".join(missing))
        else:
            entered = {**values, ROUTING_FIELD: category}
            st.session_state["result"] = {
                "category": category,
                "global": score(global_bundle, entered),
                "specialist": score(specialist, entered) if specialist else None,
            }

    result = st.session_state.get("result")
    if result and result["category"] == category:
        st.subheader("Global model")
        probability = result["global"]
        first, second = st.columns(2)
        first.metric("Predicted fraud probability", f"{probability:.1%}")
        second.metric("Score (0-100)", f"{probability * 100:.0f}")
        st.caption(
            "Models are trained with class weighting, so this output is uncalibrated: use it to rank claims, "
            "not as a literal likelihood. The recorded rate in this dataset is about 6%."
        )
        show_threshold_results(probability, global_bundle["threshold_options"])
        if specialist:
            show_specialist(result["specialist"], specialist, specialist_report["specialists"][category])
        st.caption(GLOBAL_MODEL_NOTE)


if __name__ == "__main__":
    main()
