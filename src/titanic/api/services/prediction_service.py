from src.titanic.api.services.feature_service import feature_service
from src.titanic.api.services.model_service import model_service


class PredictionService:
    def predict(self, payload):
        X = feature_service.transform_passenger(payload)
        X = X[model_service.features]

        prediction = int(model_service.predict(X)[0])
        survival_probability = float(model_service.predict_proba(X)[0][1])
        death_probability = float(1 - survival_probability)

        return {
            "prediction": prediction,
            "label": "Survived" if prediction == 1 else "Did not survive",
            "survival_probability": survival_probability,
            "death_probability": death_probability,
        }


prediction_service = PredictionService()
