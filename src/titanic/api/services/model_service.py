import joblib


class ModelService:
    def __init__(self, model_path="models/api_model.joblib"):
        bundle = joblib.load(model_path)

        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]

        self.shap_background = bundle["shap_background"]

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


model_service = ModelService()
