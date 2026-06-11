# selection.py

BASE_FEATURES = [
    "Pclass",
    "Title",
    "HasNickname",
    "SexIsMale",
    "AgeETR",
    "AgeGroup",
    "SibSp",
    "Parch",
    "GroupType",
    "FarePerPerson_log1p",
    "Deck",
    "HasCabin",
    #"EmbarkedMode",
]

FEATURE_SETS = {
    "base": BASE_FEATURES,

    "without_age_group": [
        col for col in BASE_FEATURES
        if col != "AgeGroup"
    ],

    "without_title": [
        col for col in BASE_FEATURES
        if col != "Title"
    ],

    "without_sex": [
        col for col in BASE_FEATURES
        if col != "SexIsMale"
    ],

    "without_title_and_sex": [
        col for col in BASE_FEATURES
        if col not in ["Title", "SexIsMale"]
    ],

    "without_age": [
        col for col in BASE_FEATURES
        if col not in ["AgeETR", "AgeGroup"]
    ],

    "without_fare": [
        col for col in BASE_FEATURES
        if col != "FarePerPerson_log1p"
    ],

    "without_group": [
        col for col in BASE_FEATURES
        if col != "GroupType"
    ],

    "without_cabin": [
        col for col in BASE_FEATURES
        if col not in ["Deck", "HasCabin"]
    ],

    "without_family_raw": [
        col for col in BASE_FEATURES
        if col not in ["SibSp", "Parch"]
    ],

    "minimal_strong": [
        "Pclass",
        "Title",
        "AgeETR",
        "FarePerPerson_log1p",
        "HasCabin",
    ],

    "minimal_strong_with_group": [
        "Pclass",
        "Title",
        "AgeETR",
        "FarePerPerson_log1p",
        "HasCabin",
        "GroupType",
    ],

    "with_family": BASE_FEATURES + [
        "IsAlone",
        "FamilySize",
    ],

    "with_ticket": BASE_FEATURES + [
        "TicketNumberClass",
    ],

    "with_cabin": BASE_FEATURES + [
        "CabinCount",
    ],
}


def select_model_features(
    df,
    feature_set="base",
    target="Survived",
    index_col="PassengerId",
    include_target=True,
):
    df = df.copy()

    if index_col in df.columns:
        df = df.set_index(index_col, drop=True)

    features = FEATURE_SETS[feature_set]

    selected_columns = features.copy()

    if include_target and target in df.columns:
        selected_columns = [target] + selected_columns

    missing_columns = [
        col for col in selected_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns in dataframe: {missing_columns}")

    return df[selected_columns]