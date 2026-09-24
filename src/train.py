import json
import os
import sys

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "model/diabetes_model.joblib"
METRICS_PATH = "model/metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# The candidate model must beat the baseline
# by at least this accuracy improvement.
IMPROVEMENT_MARGIN = 0.05


EXPECTED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN = "Outcome"


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs("model", exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

if not os.path.exists(DATA_PATH):
    print("ERROR: Dataset file not found:", DATA_PATH)
    sys.exit(1)

data = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Validate dataset columns
# --------------------------------------------------

print("Validating dataset...")

missing_features = [
    column
    for column in EXPECTED_FEATURES
    if column not in data.columns
]

if missing_features:
    print(
        "ERROR: Missing required feature columns:",
        missing_features
    )
    sys.exit(1)


if TARGET_COLUMN not in data.columns:
    print(
        "ERROR: Missing target column:",
        TARGET_COLUMN
    )
    sys.exit(1)


if data.empty:
    print("ERROR: Dataset is empty.")
    sys.exit(1)


print("Dataset validation passed.")
print("Number of rows:", len(data))
print("Number of features:", len(EXPECTED_FEATURES))


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

X = data[EXPECTED_FEATURES]
y = data[TARGET_COLUMN]


# --------------------------------------------------
# Train-validation split
# --------------------------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# --------------------------------------------------
# Baseline model
# --------------------------------------------------

print("\nTraining baseline...")

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_valid)

baseline_score = accuracy_score(
    y_valid,
    baseline_predictions
)


# --------------------------------------------------
# Candidate model
# --------------------------------------------------

print("Training candidate model...")

candidate = candidate = DummyClassifier(strategy="most_frequent")

candidate.fit(X_train, y_train)

candidate_predictions = candidate.predict(X_valid)

candidate_score = accuracy_score(
    y_valid,
    candidate_predictions
)


# --------------------------------------------------
# Quality gate
# --------------------------------------------------

required_score = (
    baseline_score + IMPROVEMENT_MARGIN
)

gate_passed = candidate_score >= required_score


print("\n========== RESULTS ==========")

print(
    "Baseline accuracy:",
    round(baseline_score, 4)
)

print(
    "Candidate accuracy:",
    round(candidate_score, 4)
)

print(
    "Required accuracy:",
    round(required_score, 4)
)

print(
    "Improvement margin:",
    IMPROVEMENT_MARGIN
)


if not gate_passed:

    print("\nQUALITY GATE: FAILED")

    print(
        "Candidate model did not improve enough "
        "over the baseline."
    )

    # Non-zero exit code.
    sys.exit(1)


print("\nQUALITY GATE: PASSED")


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(
    candidate,
    MODEL_PATH
)

print("\nModel saved to:", MODEL_PATH)


# --------------------------------------------------
# Save metrics report
# --------------------------------------------------

metrics = {
    "dataset": "Pima Indians Diabetes Dataset",
    "task": "binary classification",
    "target": TARGET_COLUMN,
    "metric": "accuracy",
    "random_state": RANDOM_STATE,
    "validation_size": TEST_SIZE,
    "baseline_score": baseline_score,
    "model_score": candidate_score,
    "improvement_margin": IMPROVEMENT_MARGIN,
    "required_score": required_score,
    "gate_result": "PASS",
}


with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)


print("Metrics saved to:", METRICS_PATH)

print("\nTraining completed successfully.")