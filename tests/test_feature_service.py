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
    }
    data.update(overrides)
    return PassengerInput(**data)


def test_transform_passenger_produces_all_api_model_features_without_missing_values():
    transformed = feature_service.transform_passenger(make_passenger_input())

    assert transformed.shape == (1, len(model_service.features))
    assert list(transformed.columns) == model_service.features
    assert not transformed.isna().any().any()
    assert not {
        "FamilySurvivalRate",
        "TicketSurvivalRate",
        "FarePerPerson_log1p",
        "HasNickname",
        "HasCabin",
    }.intersection(transformed.columns)


def test_transform_passenger_preserves_deterministic_input_values():
    passenger = make_passenger_input(
        pclass=1,
        title="Mrs",
        age=42.5,
        sex="female",
        travel_group_size=2,
    )

    transformed = feature_service.transform_passenger(passenger).iloc[0]

    assert transformed["Pclass"] == passenger.pclass
    assert transformed["Title"] == passenger.title
    assert transformed["SexIsMale"] == 0
    assert transformed["AgeETR"] == passenger.age
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


@pytest.mark.parametrize(("sex", "expected_sex_is_male"), [("male", 1), ("female", 0)])
def test_transform_passenger_derives_sex_is_male(sex, expected_sex_is_male):
    transformed = feature_service.transform_passenger(
        make_passenger_input(sex=sex)
    ).iloc[0]

    assert transformed["SexIsMale"] == expected_sex_is_male
