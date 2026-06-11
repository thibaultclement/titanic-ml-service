import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def split_train_test(df, target="Survived"):
    train = df[df[target].notna()].copy()
    test = df[df[target].isna()].copy()

    X_train = train.drop(columns=[target])
    y_train = train[target].astype(int)

    X_test = test.drop(columns=[target])

    return X_train, y_train, X_test


def infer_feature_types(df, features):
    X = df[features].copy()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(df, features, scale_numeric=True):
    numeric_features, categorical_features = infer_feature_types(df, features)

    if scale_numeric:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def preprocess_train_test(df, features, target="Survived", scale_numeric=True):
    X_train, y_train, X_test = split_train_test(df, target=target)

    preprocessor = build_preprocessor(
        df=X_train,
        features=features,
        scale_numeric=scale_numeric
    )

    X_train_processed = preprocessor.fit_transform(X_train[features])
    X_test_processed = preprocessor.transform(X_test[features])

    return X_train_processed, y_train, X_test_processed, preprocessor