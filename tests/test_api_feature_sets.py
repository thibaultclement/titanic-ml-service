import pandas as pd
import pytest

from titanic.selection import FEATURE_SETS, select_model_features_set

API_FEATURE_SET_NAMES = [
    "api_minimal",
    "api_with_title",
    "api_with_title_and_fare",
    "api_with_title_and_cabin",
    "api_with_title_fare_and_cabin",
]

FORBIDDEN_API_FEATURES = {
    "FamilySurvivalRate",
    "TicketSurvivalRate",
    "FarePerPerson_log1p",
    "HasNickname",
}


@pytest.mark.parametrize("feature_set_name", API_FEATURE_SET_NAMES)
def test_api_feature_sets_exclude_unavailable_or_target_derived_features(
    feature_set_name,
):
    assert not FORBIDDEN_API_FEATURES.intersection(FEATURE_SETS[feature_set_name])


@pytest.mark.parametrize("feature_set_name", API_FEATURE_SET_NAMES)
def test_api_feature_selection_preserves_the_declared_feature_order(feature_set_name):
    features = FEATURE_SETS[feature_set_name]
    df = pd.DataFrame(
        {
            "PassengerId": [1],
            "Survived": [1],
            **{feature: [0] for feature in features},
        }
    )

    selected = select_model_features_set(df, feature_set=feature_set_name)

    assert list(selected.columns) == ["Survived", *features]
