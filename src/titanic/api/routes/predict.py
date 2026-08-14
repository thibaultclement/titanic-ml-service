from fastapi import APIRouter

from src.titanic.api.schemas.input import PassengerInput
from src.titanic.api.schemas.output import PredictionOutput
from src.titanic.api.services.prediction_service import prediction_service

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
def predict(payload: PassengerInput):
    return prediction_service.predict(payload)
