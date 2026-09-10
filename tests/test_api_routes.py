from fastapi.testclient import TestClient

from titanic.api.main import app
from titanic.api.services.model_service import model_service

client = TestClient(app)

PAYLOAD = {
    "pclass": 3,
    "title": "Mr",
    "age": 28,
    "sex": "male",
    "travel_group_size": 1,
}


def test_health_endpoint_is_stable():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_describes_the_api_artifact():
    response = client.get("/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["model_name"] == "SVC"
    assert body["features"] == model_service.features
    assert body["n_features"] == 6
    assert body["api_cv_accuracy"] == 0.8058


def test_predict_endpoint_uses_the_api_feature_contract():
    response = client.post("/predict", json=PAYLOAD)
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] in {0, 1}
    assert 0 <= body["survival_probability"] <= 1
    assert 0 <= body["death_probability"] <= 1
    assert round(body["survival_probability"] + body["death_probability"], 10) == 1


def test_explain_endpoint_returns_contributions_for_api_features():
    response = client.post("/explain", json=PAYLOAD)
    body = response.json()

    assert response.status_code == 200
    assert {factor["feature"] for factor in body["top_factors"]} == set(
        model_service.features
    )
