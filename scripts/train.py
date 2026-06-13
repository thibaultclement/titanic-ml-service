# scripts/train.py

from pathlib import Path
import joblib

from titanic.data import load_processed_data
from titanic.selection import select_model_features
from titanic.models import get_final_model

from titanic.preprocessing import (
    preprocess_train_test,
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_BINARY_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
)


def main():
    df = load_processed_data()

    df = select_model_features(
        df,
        include_target=True
    )

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

    model = get_final_model(random_state=42)

    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "model_name": model.__class__.__name__,
            "preprocessor": preprocessor,
            "features": features,
        },
        "models/model.joblib"
    )

    print(f"Final model trained: {model.__class__.__name__}")
    print(f"Features used: {features}")
    print("Model saved to models/model.joblib")


if __name__ == "__main__":
    main()