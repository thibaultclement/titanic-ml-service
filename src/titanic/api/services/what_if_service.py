from src.titanic.api.services.prediction_service import prediction_service


class WhatIfService:
    def compare(self, payload):
        before = prediction_service.predict(payload.before)
        after = prediction_service.predict(payload.after)

        return {
            "before": before,
            "after": after,
            "probability_difference": (
                after["survival_probability"]
                - before["survival_probability"]
            ),
        }


what_if_service = WhatIfService()