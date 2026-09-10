import pandas as pd

from titanic.api.services.model_service import model_service


class FeatureService:
    def is_child(self, age: float) -> int:
        return int(age < 7)

    def sex_is_male(self, sex: str) -> int:
        return int(sex == "male")

    def transform_passenger(self, payload) -> pd.DataFrame:
        data = payload.model_dump()

        features = {
            "Pclass": data["pclass"],
            "Title": data["title"],
            "SexIsMale": self.sex_is_male(data["sex"]),
            "AgeETR": data["age"],
            "IsChild": self.is_child(data["age"]),
            "IsAlone": int(data["travel_group_size"] == 1),
        }

        return pd.DataFrame([features], columns=model_service.features)


feature_service = FeatureService()
