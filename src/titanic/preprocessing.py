import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


DEFAULT_CATEGORICAL_FEATURES = [
    "Pclass",
    "Title",
    "AgeGroup",
    "AgeDecade",
    "GroupType",
    "Deck",
    "Embarked",
    "EmbarkedMode",
]

DEFAULT_BINARY_FEATURES = [
    "SexIsMale",
    "HasNickname",
    "IsAlone",
    "HasFamily",
    "HasCabin",
    "IsChild",
]

DEFAULT_NUMERIC_FEATURES = [
    "AgeETR",
    "SibSp",
    "Parch",
    "FamilySize",
    "GroupSize",
    "FarePerPerson_log1p",
    "FarePerTicketPassenger_log1p",
    "CabinNumber",
    "CabinCount",
    "TicketNumberClass",
]


def split_train_test(df, target="Survived"):
    train = df[df[target].notna()].copy()
    test = df[df[target].isna()].copy()

    X_train = train.drop(columns=[target])
    y_train = train[target].astype(int)

    X_test = test.drop(columns=[target])

    return X_train, y_train, X_test


def infer_feature_types(
    df,
    features,
    categorical_features=None,
    binary_features=None,
    numeric_features=None,
):
    features = list(features)

    categorical_features = categorical_features or []
    binary_features = binary_features or []
    numeric_features = numeric_features or []

    categorical_features = [
        col for col in categorical_features
        if col in features
    ]

    binary_features = [
        col for col in binary_features
        if col in features
    ]

    numeric_features = [
        col for col in numeric_features
        if col in features
    ]

    already_assigned = set(
        categorical_features
        + binary_features
        + numeric_features
    )

    remaining_features = [
        col for col in features
        if col not in already_assigned
    ]

    inferred_categorical = (
        df[remaining_features]
        .select_dtypes(include=["object", "category", "bool"])
        .columns
        .tolist()
    )

    inferred_numeric = (
        df[remaining_features]
        .select_dtypes(include=["int64", "float64", "int32", "float32"])
        .columns
        .tolist()
    )

    categorical_features = categorical_features + inferred_categorical
    numeric_features = numeric_features + inferred_numeric

    return numeric_features, categorical_features, binary_features


def build_preprocessor(
    df,
    features,
    scale_numeric=True,
    categorical_features=None,
    binary_features=None,
    numeric_features=None,
):
    numeric_features, categorical_features, binary_features = infer_feature_types(
        df=df,
        features=features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        numeric_features=numeric_features,
    )

    if scale_numeric:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features),
        ],
        remainder="drop",
    )

    return preprocessor


def preprocess_train_test(
    df,
    features,
    target="Survived",
    scale_numeric=True,
    categorical_features=None,
    binary_features=None,
    numeric_features=None,
):
    X_train, y_train, X_test = split_train_test(df, target=target)

    preprocessor = build_preprocessor(
        df=X_train,
        features=features,
        scale_numeric=scale_numeric,
        categorical_features=categorical_features,
        binary_features=binary_features,
        numeric_features=numeric_features,
    )

    X_train_processed = preprocessor.fit_transform(X_train[features])
    X_test_processed = preprocessor.transform(X_test[features])

    return X_train_processed, y_train, X_test_processed, preprocessor