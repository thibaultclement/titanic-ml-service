from fastapi import APIRouter

from src.titanic.api.schemas.input import PassengerInput
from src.titanic.api.schemas.output import ExplainOutput
from src.titanic.api.services.explainability_service import explainability_service

router = APIRouter()


@router.post("/explain", response_model=ExplainOutput)
def explain(payload: PassengerInput):
    return explainability_service.explain(payload)
