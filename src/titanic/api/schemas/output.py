from pydantic import BaseModel


class PredictionOutput(BaseModel):
    prediction: int
    label: str
    survival_probability: float
    death_probability: float


class WhatIfOutput(BaseModel):
    before: PredictionOutput
    after: PredictionOutput
    probability_difference: float


class FactorContribution(BaseModel):
    feature: str
    contribution: float
    direction: str


class ExplainOutput(BaseModel):
    prediction: int
    label: str
    survival_probability: float
    death_probability: float
    top_factors: list[FactorContribution]