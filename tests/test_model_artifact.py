from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

EXPECTED_FEATURES = [
    "Pclass",
    "Title",
    "HasNickname",
    "AgeETR",
    "IsChild",
    "IsAlone",
    "FamilySurvivalRate",
    "FarePerPerson_log1p",
    "TicketSurvivalRate",
    "HasCabin",
]

REQUIRED_BUNDLE_KEYS = {
    "pipeline",
    "model_name",
    "features",
    "global_survival_rate",
    "fare_per_person_by_pclass",
    "global_fare_per_person_log1p",
    "shap_background",
}


@pytest.fixture(scope="module")
def model_bundle():
    assert MODEL_PATH.is_file(), f"Model artifact not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def test_model_bundle_contains_required_contract_keys(model_bundle):
    assert REQUIRED_BUNDLE_KEYS <= model_bundle.keys()


def test_model_bundle_features_match_frozen_contract(model_bundle):
    assert model_bundle["features"] == EXPECTED_FEATURES


def test_model_bundle_pipeline_matches_feature_contract(model_bundle):
    pipeline = model_bundle["pipeline"]

    assert set(pipeline.named_steps) == {"preprocessor", "model"}
    assert pipeline.n_features_in_ == len(EXPECTED_FEATURES)
    assert model_bundle["model_name"] == type(pipeline.named_steps["model"]).__name__


def test_model_bundle_reference_metadata_is_usable(model_bundle):
    background = model_bundle["shap_background"]

    assert not background.empty
    assert list(background.columns) == EXPECTED_FEATURES
    assert isinstance(model_bundle["global_survival_rate"], Real)
    assert isinstance(model_bundle["global_fare_per_person_log1p"], Real)
    assert isinstance(model_bundle["fare_per_person_by_pclass"], dict)


def test_model_bundle_predicts_valid_probabilities_from_reference_row(model_bundle):
    pipeline = model_bundle["pipeline"]
    reference_row = model_bundle["shap_background"].loc[:, EXPECTED_FEATURES].head(1)

    prediction = pipeline.predict(reference_row)
    probabilities = pipeline.predict_proba(reference_row)

    assert prediction.shape == (1,)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
