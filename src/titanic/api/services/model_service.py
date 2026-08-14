import joblib


class ModelService:
    def __init__(self, model_path="models/model.joblib"):
        bundle = joblib.load(model_path)

        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]

        self.global_survival_rate = bundle["global_survival_rate"]
        self.fare_per_person_by_pclass = bundle["fare_per_person_by_pclass"]
        self.global_fare_per_person_log1p = bundle["global_fare_per_person_log1p"]

        self.shap_background = bundle["shap_background"]

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


model_service = ModelService()
