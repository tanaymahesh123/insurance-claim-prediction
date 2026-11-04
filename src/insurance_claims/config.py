"""Paths, observed schema, and feature exclusions for the raw CSV."""

from pathlib import Path

DEFAULT_DATA_PATH = Path("data/raw/fraud_oracle.csv")
TARGET = "FraudFound_P"  # integer 0/1
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

EXPECTED_COLUMNS = (
    "Month",
    "WeekOfMonth",
    "DayOfWeek",
    "Make",
    "AccidentArea",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "WeekOfMonthClaimed",
    "Sex",
    "MaritalStatus",
    "Age",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "VehiclePrice",
    "FraudFound_P",
    "PolicyNumber",
    "RepNumber",
    "Deductible",
    "DriverRating",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "AgeOfPolicyHolder",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "Year",
    "BasePolicy",
)

# Identifiers — no predictive value, would cause overfitting
EXCLUDED_IDENTIFIERS = ["PolicyNumber", "RepNumber"]

# Demographic fields — excluded for fairness/legal reasons
# Using Sex or MaritalStatus in fraud scoring raises discrimination concerns
EXCLUDED_DEMOGRAPHIC = ["Sex", "MaritalStatus"]

# Post-claim timing fields — may not be available at prediction time
EXCLUDED_POST_CLAIM = []  # none confirmed post-claim in this dataset yet

EXCLUDED_ALWAYS = EXCLUDED_IDENTIFIERS + EXCLUDED_DEMOGRAPHIC

# Read by the data-quality check, which reports excluded and date columns.
FEATURE_EXCLUSIONS = frozenset(EXCLUDED_ALWAYS)
DATE_COLUMNS = ()

# Model inputs, including columns added by preprocessing.engineer_features().
FEATURES = [
    # NOTE: Age and AgeOfPolicyHolder may raise age-discrimination concerns
    # in a real deployment. Review with legal/compliance before production use.
    "Age",
    "Deductible",
    "DriverRating",
    "WeekOfMonth",
    "WeekOfMonthClaimed",
    "Year",
    "VehiclePrice",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    # NOTE: Age and AgeOfPolicyHolder may raise age-discrimination concerns
    # in a real deployment. Review with legal/compliance before production use.
    "AgeOfPolicyHolder",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "fault_policy_holder",
    "no_police_report",
    "no_witness",
    "high_driver_rating",
    "address_changed_recently",
    "Make",
    "AccidentArea",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "BasePolicy",
    "Month",
    "DayOfWeek",
    "MonthClaimed",
    "DayOfWeekClaimed",
]

# Claim-timing fields kept as inputs but flagged for review. If they turn out not
# to be known at the intended prediction moment, move them to EXCLUDED_POST_CLAIM.
FRAUD_AVAILABILITY_REVIEW_FIELDS = (
    "WeekOfMonthClaimed",
    "MonthClaimed",
    "DayOfWeekClaimed",
    "Days_Policy_Claim",
)
