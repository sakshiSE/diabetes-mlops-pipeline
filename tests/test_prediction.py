import numpy as np
import pytest

from src.predict import load_model, predict


VALID_SAMPLE = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 30.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35,
}


def test_model_can_be_loaded():
    model = load_model()

    assert model is not None


def test_valid_prediction():
    result = predict(VALID_SAMPLE)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[0] in [0, 1]


def test_missing_feature_is_rejected():
    invalid_sample = VALID_SAMPLE.copy()

    del invalid_sample["Glucose"]

    with pytest.raises(ValueError, match="Missing required feature"):
        predict(invalid_sample)