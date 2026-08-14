import shap
import pandas as pd

from src.titanic.api.services.feature_service import feature_service
from src.titanic.api.services.model_service import model_service
from src.titanic.api.services.prediction_service import prediction_service


class ExplainabilityService:
    def __init__(self):
        self.feature_names = model_service.features
        self.background = model_service.shap_background

        def predict_fn(X):
            X = pd.DataFrame(
                X,
                columns=self.feature_names,
            )

            return model_service.predict_proba(X)[:, 1]

        self.explainer = shap.KernelExplainer(
            predict_fn,
            self.background,
        )

    def explain(self, payload, top_n=None):
        prediction = prediction_service.predict(payload)

        X = feature_service.transform_passenger(payload)
        X = X[model_service.features]

        shap_values = self.explainer.shap_values(
            X,
            nsamples=100,
        )

        contributions = []

        for feature, contribution in zip(
            self.feature_names,
            shap_values[0],
            strict=False,
        ):
            contribution = float(contribution)

            contributions.append({
                "feature": feature,
                "contribution": contribution,
                "direction": "positive" if contribution >= 0 else "negative",
            })

        contributions = sorted(
            contributions,
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )

        return {
            **prediction,
            "top_factors": contributions,
        }


explainability_service = ExplainabilityService()
