# Insurance Fraud Label Prediction

An educational project that predicts the recorded `FraudFound_P` label on vehicle insurance claims. It covers a reproducible train/validation/test split, a logistic regression baseline, a validation ranking of six models, a Sedan specialist, and a local Streamlit demo.

The results are exploratory. They do not establish confirmed fraud or real-world fraud detection.

## Dataset

| | |
|---|---|
| File | `fraud_oracle.csv`, copied to `data/raw/fraud_oracle.csv` |
| Size | 15,420 rows, 33 columns |
| Target | `FraudFound_P` (0 or 1) |
| Positive rate | 923 rows, about 6% |
| Split | Stratified 70/15/15, seed 42: 10,794 train, 2,313 validation, 2,313 test |

`PolicyNumber` and `RepNumber` are excluded as identifiers. `Sex` and `MaritalStatus` are excluded as demographic fields.

## Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw && cp fraud_oracle.csv data/raw/fraud_oracle.csv
```

## How to run

Run these in order from the project root.

1. Prepare the split and fit preprocessing on the training rows:

   ```bash
   PYTHONPATH=src python -m insurance_claims.prepare_cli
   ```

2. Train the logistic regression baseline. This is the only step that reports test metrics:

   ```bash
   PYTHONPATH=src python -m insurance_claims.train_cli
   ```

3. Compare the six models on the validation split:

   ```bash
   PYTHONPATH=src python -m insurance_claims.feature_set_compare_cli
   ```

4. Train the vehicle-category specialists, which the app needs:

   ```bash
   PYTHONPATH=src python -m insurance_claims.incident_specialists_cli
   ```

5. Start the demo:

   ```bash
   streamlit run app.py --server.address 127.0.0.1 --server.port 8501
   ```

Outputs are written to `data/processed/`. Superseded outputs from earlier experiments are kept in `data/processed/archive/`. The saved models use Joblib serialization, so only load artifacts you trust.

## Final ranking

Validation results, sorted by average precision (AP). The test split was not used.

| Model | Features | Val AP | Val F1 | Val Precision | Val Recall | FP count |
|---|---|---:|---:|---:|---:|---:|
| `reduced_catboost` | `no_claim_timing_29` | 0.2593 | 0.3256 | 0.2540 | 0.4532 | 185 |
| `full_catboost` | `full_33` | 0.2538 | 0.3126 | 0.2167 | 0.5612 | 282 |
| `full_xgboost` | `full_33` | 0.2511 | 0.3059 | 0.2083 | 0.5755 | 304 |
| `stacking_xgb_rf` | `full_xgboost+reduced_random_forest` | 0.2418 | 0.3096 | 0.2392 | 0.4388 | 194 |
| `reduced_xgboost_shap_pruned` | `shap_pruned_24` | 0.2398 | 0.3048 | 0.2278 | 0.4604 | 217 |
| `reduced_xgboost` | `no_claim_timing_29` | 0.2382 | 0.3182 | 0.2232 | 0.5540 | 268 |
| `reduced_random_forest` | `no_claim_timing_29` | 0.2169 | 0.2763 | 0.1818 | 0.5755 | 360 |
| `full_random_forest` | `full_33` | 0.1840 | 0.2710 | 0.1730 | 0.6259 | 416 |
| `reduced_logistic` | `no_claim_timing_29` | 0.1499 | 0.2540 | 0.1629 | 0.5755 | 411 |
| `full_logistic` | `full_33` | 0.1481 | 0.2587 | 0.1681 | 0.5612 | 386 |

F1, precision, recall, and FP count use each model's max-F1 threshold. A random ranking would score an AP near the 6% positive rate. The source is `data/processed/fraud_validation_model_ranking.json`.

The logistic baseline is the only model scored on the test split. It reached AP 0.1508 and ROC AUC 0.7925.

Nested cross-validation on the training rows gives the logistic baseline an AP of 0.1437 ± 0.0132 and an F1 of 0.2217 ± 0.0194. AP does not depend on the threshold and the baseline has no tuned hyperparameters, so this AP equals the plain 5-fold result. It says nothing about the tree models.

## SHAP feature importance

Mean absolute SHAP values for `full_xgboost` on the validation rows. The top 15 are saved in `data/processed/fraud_shap_importance.json`.

| Feature | Mean \|SHAP\| |
|---|---:|
| `fault_policy_holder` | 1.0248 |
| `BasePolicy` | 0.7865 |
| `Age` | 0.3506 |
| `Fault` | 0.2923 |
| `Month` | 0.2674 |
| `Year` | 0.2355 |
| `PolicyType` | 0.2188 |
| `MonthClaimed` | 0.1969 |
| `Make` | 0.1902 |
| `Deductible` | 0.1717 |
| `DayOfWeek` | 0.1576 |
| `DayOfWeekClaimed` | 0.1518 |
| `DriverRating` | 0.1379 |
| `WeekOfMonth` | 0.1350 |
| `WeekOfMonthClaimed` | 0.1347 |

`fault_policy_holder` and `Fault` are duplicates, so their credit is shared between them. `reduced_xgboost_shap_pruned` removes the five `reduced_xgboost` inputs in the bottom 20%: `Days_Policy_Accident`, `no_witness`, `address_changed_recently`, `PoliceReportFiled`, and `WitnessPresent`. Its AP moved from 0.2382 to 0.2398, which is too small to read as an improvement, and those features were chosen on the same rows used to score it.

## Demo app

The app uses `reduced_xgboost` as its global model. `full_xgboost` ranks higher on validation AP, but it relies on claim-timing fields whose availability at prediction time is unverified. You choose `VehicleCategory`, enter the required fields, and get an output for the claim. You can switch between four thresholds chosen on the validation split:

| Option | `reduced_xgboost` threshold | Val precision | Val recall | Val rows flagged |
|---|---:|---:|---:|---:|
| `max_f1` | 0.5006 | 0.2232 | 0.5540 | 345 |
| `recall_70` | 0.3291 | 0.1681 | 0.7050 | 583 |
| `precision_30` | 0.7583 | 0.3000 | 0.1727 | 80 |
| `Cost-optimal` | 0.0523 | 0.1214 | 0.9784 | 1,120 |

The cost-optimal threshold minimizes 50 × false negatives + 1 × false positives on the validation split. It flags 48% of validation rows.

The output is uncalibrated because the models use class weighting. Use it to rank claims, not as a literal likelihood.

## Known issues

- **`Age = 0`:** 320 rows have Age=0, corresponding to AgeOfPolicyHolder='16 to 17'. This appears to be a coding convention and is left as-is.
- **`"0"` claim dates:** one row has `"0"` in both `DayOfWeekClaimed` and `MonthClaimed`. It is left as-is in training, and the app does not offer `"0"` as an input.
- **Only a Sedan specialist:** Sport has 53 positive training rows and Utility has 36, both below the floor of 100. Both categories use the global model.
- **Age fairness:** `Age` and `AgeOfPolicyHolder` may raise age-discrimination concerns in a real deployment. Review them with legal/compliance before production use.
- **Claim-timing fields:** `WeekOfMonthClaimed`, `MonthClaimed`, `DayOfWeekClaimed`, and `Days_Policy_Claim` may not be available at prediction time. The top-ranked `full_xgboost` relies on them. `reduced_xgboost` omits them, so the app uses it and it is the preferred model for deployment. `reduced_catboost` also omits them and currently scores higher on validation AP, but the app has not been switched to it.
- **Filtered `RuntimeWarning`s in logistic regression:** on an arm64 macOS 15 setup (numpy 2.1.3 linked to Apple Accelerate), logistic fits raise divide-by-zero, overflow, and invalid-value warnings from `matmul`. They also occur on clean random data and did not change the results: scores match a non-BLAS calculation to about 1e-15, so they appear to be spurious floating-point flags. `src/insurance_claims/__init__.py` hides only these three `matmul` messages, only on arm64 macOS. The filter would also hide a real `matmul` fault on that platform, so remove it when debugging numeric problems.
- **Redundant inputs:** `fault_policy_holder`, `no_police_report`, and `no_witness` duplicate `Fault`, `PoliceReportFiled`, and `WitnessPresent`, so individual logistic coefficients for those pairs are not meaningful.

## Limitations

- **Label meaning is undocumented:** the models predict the recorded `FraudFound_P` value, not confirmed fraud.
- **Validation scores are optimistic:** models and thresholds are chosen and scored on the same validation rows. The gap between the top two models is small and could reverse with a different split.
- **Random split:** rows are assumed to be independent. A time-based split may better match real use.
