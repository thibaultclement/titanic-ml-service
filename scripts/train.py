# scripts/train.py

from pathlib import Path
import joblib

from titanic.data import load_processed_data
from titanic.selection import select_model_features
from titanic.preprocessing import preprocess_train_test

from titanic.models import get_baseline_models
from titanic.validation import benchmark_models


def main():

    df = load_processed_data()

    df = select_model_features(
        df,
        feature_set="base"
    )

    features = [
        col
        for col in df.columns
        if col != "Survived"
    ]

    X_train, y_train, _, preprocessor = (
        preprocess_train_test(
            df,
            features=features
        )
    )

    models = get_baseline_models()

    results = benchmark_models(
        models=models,
        X=X_train,
        y=y_train
    )

    print(results)

    best_model_name = results.iloc[0]["model"]

    best_model = models[best_model_name]

    best_model.fit(
        X=X_train,
        y=y_train
    )

    Path("models").mkdir(
        exist_ok=True
    )

    joblib.dump(
        {
            "model": best_model,
            "preprocessor": preprocessor,
            "features": features
        },
        "models/model.joblib"
    )


if __name__ == "__main__":
    main()