import pandas as pd

from src.titanic.api.services.model_service import model_service


class FeatureService:
    def estimate_fare_per_person_log1p(self, pclass: int) -> float:
        return float(
            model_service.fare_per_person_by_pclass.get(
                pclass,
                model_service.global_fare_per_person_log1p,
            )
        )

    def is_child(self, age: float) -> int:
        return int(age < 7)

    def transform_passenger(self, payload) -> pd.DataFrame:
        data = payload.model_dump()

        features = {
            "Pclass": data["pclass"],
            "Title": data["title"],
            "HasNickname": 0,
            "AgeETR": data["age"],
            "IsChild": self.is_child(data["age"]),
            "FarePerPerson_log1p": self.estimate_fare_per_person_log1p(
                data["pclass"]
            ),
            "FamilySurvivalRate": float(
                model_service.global_survival_rate
            ),
            "TicketSurvivalRate": float(
                model_service.global_survival_rate
            ),
            "HasCabin": int(data["has_cabin"]),
            "IsAlone": int(data["travel_group_size"] == 1),
        }

        return pd.DataFrame([features])


feature_service = FeatureService()
