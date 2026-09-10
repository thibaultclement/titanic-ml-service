from pathlib import Path

import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "api_model.joblib"

EXPECTED_FEATURES = [
    "Pclass",
    "Title",
    "SexIsMale",
    "AgeETR",
    "IsChild",
    "IsAlone",
]

REQUIRED_BUNDLE_KEYS = {
    "pipeline",
    "model_name",
    "features",
    "shap_background",
}


@pytest.fixture(scope="module")
def model_bundle():
    assert MODEL_PATH.is_file(), f"API model artifact not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def test_api_model_bundle_contains_required_contract_keys(model_bundle):
    assert REQUIRED_BUNDLE_KEYS <= model_bundle.keys()


def test_api_model_bundle_features_match_api_contract(model_bundle):
    assert model_bundle["features"] == EXPECTED_FEATURES


def test_api_model_bundle_contains_a_predictive_pipeline(model_bundle):
    pipeline = model_bundle["pipeline"]

    assert set(pipeline.named_steps) == {"preprocessor", "model"}
    assert pipeline.n_features_in_ == len(EXPECTED_FEATURES)
    assert model_bundle["model_name"] == type(pipeline.named_steps["model"]).__name__


def test_api_model_background_matches_api_features(model_bundle):
    background = model_bundle["shap_background"]

    assert not background.empty
    assert list(background.columns) == EXPECTED_FEATURES


def test_api_model_predicts_valid_probabilities_from_reference_row(model_bundle):
    pipeline = model_bundle["pipeline"]
    reference_row = model_bundle["shap_background"].loc[:, EXPECTED_FEATURES].head(1)

    prediction = pipeline.predict(reference_row)
    probabilities = pipeline.predict_proba(reference_row)

    assert prediction.shape == (1,)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
