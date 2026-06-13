# scripts/train.py

from pathlib import Path
import joblib

from titanic.data import load_processed_data
from titanic.selection import select_model_features
from titanic.models import get_baseline_models
from titanic.validation import benchmark_models

from titanic.preprocessing import (
    preprocess_train_test,
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_BINARY_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
)


def main():
    df = load_processed_data()

    #df = select_model_features(
    #    df,
    #    feature_set="base"
    #)

    df = select_model_features(df)

    features = [
        col for col in df.columns
        if col != "Survived"
    ]

    X_train, y_train, X_test, preprocessor = preprocess_train_test(
        df,
        features=features,
        target="Survived",
        scale_numeric=True,
        categorical_features=DEFAULT_CATEGORICAL_FEATURES,
        binary_features=DEFAULT_BINARY_FEATURES,
        numeric_features=DEFAULT_NUMERIC_FEATURES,
    )

    models = get_baseline_models()

    results = benchmark_models(
        models=models,
        X=X_train,
        y=y_train
    )

    print(results)

    best_model_name = results.iloc[0]["model"]
    best_cv_score = results.iloc[0]["test_score_mean"]

    print(f"\nBest model: {best_model_name}")
    print(f"Best CV score: {best_cv_score:.4f}")

    best_model = models[best_model_name]

    best_model.fit(
        X=X_train,
        y=y_train
    )

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "cv_score": best_cv_score,
            "cv_results": results,
            "preprocessor": preprocessor,
            "features": features,
        },
        "models/model.joblib"
    )

    print("\nModel saved to models/model.joblib")


if __name__ == "__main__":
    main()