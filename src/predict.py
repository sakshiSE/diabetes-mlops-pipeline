import joblib
import pandas as pd


MODEL_PATH = "model/diabetes_model.joblib"

REQUIRED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_model():
    """Load the trained model."""

    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return model


def predict(input_data):
    """
    Make a diabetes prediction.

    input_data can be a dictionary containing
    the required feature names.
    """

    # Check that all required features exist.
    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(
            "Missing required feature(s): "
            + ", ".join(missing_features)
        )

    # Keep only the expected features
    # and keep them in the correct order.
    input_row = {
        feature: input_data[feature]
        for feature in REQUIRED_FEATURES
    }

    input_df = pd.DataFrame([input_row])

    model = load_model()

    prediction = model.predict(input_df)

    return prediction


if __name__ == "__main__":

    sample = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 30.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35,
    }

    result = predict(sample)

    print("Prediction:", result)
    print("Prediction type:", type(result))
    print("Prediction shape:", result.shape)