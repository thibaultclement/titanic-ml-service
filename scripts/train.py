from pathlib import Path
import joblib

from titanic.data import load_processed_data
from titanic.selection import select_model_features
from titanic.models import get_final_model, build_model_pipeline
from titanic.preprocessing import (
    build_preprocessor,
    get_train_test_data,
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

    X_train, y_train, X_test = get_train_test_data(
        df,
        features=features,
        target="Survived"
    )

    preprocessor = build_preprocessor(
        df=X_train,
        features=features,
        scale_numeric=True,
        categorical_features=DEFAULT_CATEGORICAL_FEATURES,
        binary_features=DEFAULT_BINARY_FEATURES,
        numeric_features=DEFAULT_NUMERIC_FEATURES,
    )

    model = get_final_model()

    pipeline = build_model_pipeline(
        preprocessor=preprocessor,
        model=model
    )

    pipeline.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": model.__class__.__name__,
            "features": features,
        },
        "models/model.joblib"
    )

    print(f"Final model trained: {model.__class__.__name__}")
    print(f"Features used: {features}")
    print("Model saved to models/model.joblib")


if __name__ == "__main__":
    main()