import pandas as pd
from sklearn.linear_model import LogisticRegression

from titanic.data import load_processed_data
from titanic.models import get_final_model
from titanic.preprocessing import (
    DEFAULT_BINARY_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    build_preprocessor,
    get_train_test_data,
)
from titanic.selection import FEATURE_SETS, select_model_features_set
from titanic.validation import benchmark_models

API_FEATURE_SET_NAMES = [
    "api_minimal",
    "api_with_title",
    "api_with_title_and_fare",
    "api_with_title_and_cabin",
    "api_with_title_fare_and_cabin",
]


def evaluate_api_feature_sets():
    df = load_processed_data()
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVC_RBF": get_final_model(),
    }
    results = []

    for feature_set_name in API_FEATURE_SET_NAMES:
        df_model = select_model_features_set(
            df,
            feature_set=feature_set_name,
            include_target=True,
        )
        features = FEATURE_SETS[feature_set_name]
        X_train, y_train, _ = get_train_test_data(df_model, features=features)

        preprocessor = build_preprocessor(
            df=X_train,
            features=features,
            scale_numeric=True,
            categorical_features=DEFAULT_CATEGORICAL_FEATURES,
            binary_features=DEFAULT_BINARY_FEATURES,
            numeric_features=DEFAULT_NUMERIC_FEATURES,
        )

        cv_results = benchmark_models(
            models=models,
            preprocessor=preprocessor,
            X=X_train,
            y=y_train,
            scoring="accuracy",
            n_splits=10,
            random_state=42,
        )
        cv_results.insert(0, "feature_set", feature_set_name)
        cv_results.insert(1, "n_features", len(features))
        cv_results.insert(2, "features", ", ".join(features))
        results.append(cv_results)

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    results = evaluate_api_feature_sets()
    print(results.to_string(index=False))
