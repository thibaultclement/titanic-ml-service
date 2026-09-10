import pandas as pd
import pytest

from titanic.api.schemas.input import PassengerInput
from titanic.api.services.feature_service import feature_service
from titanic.api.services.model_service import model_service


def make_passenger_input(**overrides) -> PassengerInput:
    data = {
        "pclass": 3,
        "title": "Mr",
        "age": 28,
        "sex": "male",
        "travel_group_size": 1,
        "has_cabin": False,
    }
    data.update(overrides)
    return PassengerInput(**data)


def test_transform_passenger_produces_all_model_features_without_missing_values():
    transformed = feature_service.transform_passenger(make_passenger_input())

    assert transformed.shape == (1, len(model_service.features))
    assert set(transformed.columns) == set(model_service.features)
    assert not transformed.isna().any().any()
    assert transformed.loc[0, "HasCabin"] == 0

    reindexed = transformed.loc[:, model_service.features]

    assert list(reindexed.columns) == model_service.features


def test_transform_passenger_preserves_deterministic_input_values():
    passenger = make_passenger_input(
        pclass=1,
        title="Mrs",
        age=42.5,
        travel_group_size=2,
        has_cabin=True,
    )

    transformed = feature_service.transform_passenger(passenger).iloc[0]

    assert transformed["Pclass"] == passenger.pclass
    assert transformed["Title"] == passenger.title
    assert transformed["AgeETR"] == passenger.age
    assert transformed["HasNickname"] == 0
    assert transformed["HasCabin"] == 1
    assert transformed["IsAlone"] == 0


@pytest.mark.parametrize(
    ("age", "expected_is_child"),
    [(6.99, 1), (7.0, 0)],
)
def test_transform_passenger_derives_is_child_from_age(age, expected_is_child):
    transformed = feature_service.transform_passenger(
        make_passenger_input(age=age)
    ).iloc[0]

    assert transformed["IsChild"] == expected_is_child


@pytest.mark.parametrize(
    ("travel_group_size", "expected_is_alone"),
    [(1, 1), (2, 0), (20, 0)],
)
def test_transform_passenger_derives_is_alone_from_group_size(
    travel_group_size, expected_is_alone
):
    transformed = feature_service.transform_passenger(
        make_passenger_input(travel_group_size=travel_group_size)
    ).iloc[0]

    assert transformed["IsAlone"] == expected_is_alone


def test_transform_passenger_currently_does_not_use_sex():
    male_features = feature_service.transform_passenger(
        make_passenger_input(sex="male")
    )
    female_features = feature_service.transform_passenger(
        make_passenger_input(sex="female")
    )

    pd.testing.assert_frame_equal(male_features, female_features)


def test_transform_passenger_uses_global_rate_for_group_survival_features():
    transformed = feature_service.transform_passenger(make_passenger_input()).iloc[0]

    assert transformed["FamilySurvivalRate"] == model_service.global_survival_rate
    assert transformed["TicketSurvivalRate"] == model_service.global_survival_rate


@pytest.mark.parametrize("pclass", [1, 2, 3])
def test_transform_passenger_uses_pclass_fare_reference(pclass):
    transformed = feature_service.transform_passenger(
        make_passenger_input(pclass=pclass)
    ).iloc[0]

    assert transformed["FarePerPerson_log1p"] == (
        model_service.fare_per_person_by_pclass[pclass]
    )


def test_fare_estimate_falls_back_to_global_reference_for_unknown_pclass():
    assert feature_service.estimate_fare_per_person_log1p(999) == (
        model_service.global_fare_per_person_log1p
    )
