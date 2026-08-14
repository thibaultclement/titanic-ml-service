from fastapi import APIRouter

from src.titanic.api.schemas.input import WhatIfInput
from src.titanic.api.schemas.output import WhatIfOutput
from src.titanic.api.services.what_if_service import what_if_service

router = APIRouter()


@router.post("/what-if", response_model=WhatIfOutput)
def what_if(payload: WhatIfInput):
    return what_if_service.compare(payload)
