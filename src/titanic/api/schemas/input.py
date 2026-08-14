from typing import Literal

from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    pclass: Literal[1, 2, 3] = Field(description="Passenger class: 1, 2 or 3.")
    title: Literal["Mr", "Mrs", "Miss", "Master"] = Field(
        description="Passenger title."
    )
    age: float = Field(default=28.0, ge=0, le=100, description="Passenger age.")
    sex: Literal["male", "female"] = Field(description="Passenger sex.")
    travel_group_size: int = Field(
        default=1, ge=1, le=20, description="Number of people travelling together."
    )
    has_cabin: bool = Field(
        default=False, description="Whether the passenger has a known cabin."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "pclass": 3,
                "title": "Mr",
                "age": 28,
                "sex": "male",
                "travel_group_size": 1,
                "has_cabin": False,
            }
        }
    }


class WhatIfInput(BaseModel):
    before: PassengerInput
    after: PassengerInput

    model_config = {
        "json_schema_extra": {
            "example": {
                "before": {
                    "pclass": 3,
                    "title": "Mr",
                    "age": 28,
                    "sex": "male",
                    "travel_group_size": 1,
                    "has_cabin": False,
                },
                "after": {
                    "pclass": 1,
                    "title": "Mr",
                    "age": 28,
                    "sex": "male",
                    "travel_group_size": 1,
                    "has_cabin": True,
                },
            }
        }
    }
