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
    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = load_processed_data()

    df = select_model_features(
        df,
        include_target=True,
    )

    features = [
        col
        for col in df.columns
        if col != "Survived"
    ]

    # ---------------------------------------------------------
    # Train / test split
    # ---------------------------------------------------------

    X_train, y_train, X_test = get_train_test_data(
        df,
        features=features,
        target="Survived",
    )

    # ---------------------------------------------------------
    # Production metadata
    # ---------------------------------------------------------

    global_survival_rate = float(y_train.mean())

    fare_per_person_by_pclass = (
        X_train
        .groupby("Pclass")["FarePerPerson_log1p"]
        .median()
        .to_dict()
    )

    global_fare_per_person_log1p = float(
        X_train["FarePerPerson_log1p"].median()
    )

    shap_background = (
        X_train[features]
        .sample(
            n=min(100, len(X_train)),
            random_state=42,
        )
        .copy()
    )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    preprocessor = build_preprocessor(
        df=X_train,
        features=features,
        scale_numeric=True,
        categorical_features=DEFAULT_CATEGORICAL_FEATURES,
        binary_features=DEFAULT_BINARY_FEATURES,
        numeric_features=DEFAULT_NUMERIC_FEATURES,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = get_final_model()

    pipeline = build_model_pipeline(
        preprocessor=preprocessor,
        model=model,
    )

    pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Save production artifact
    # ---------------------------------------------------------

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": model.__class__.__name__,
            "features": features,

            # Reference values used by the API
            "global_survival_rate": global_survival_rate,
            "fare_per_person_by_pclass": fare_per_person_by_pclass,
            "global_fare_per_person_log1p": global_fare_per_person_log1p,

            # Reference sample used by SHAP
            "shap_background": shap_background,
        },
        "models/model.joblib",
    )

    print(f"Final model trained: {model.__class__.__name__}")
    print(f"Features used: {features}")
    print("Model saved to models/model.joblib")


if __name__ == "__main__":
    main()
