import argparse
from pathlib import Path

import joblib

from titanic.data import load_processed_data
from titanic.models import build_model_pipeline, get_final_model
from titanic.preprocessing import (
    DEFAULT_BINARY_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    build_preprocessor,
    get_train_test_data,
)
from titanic.selection import select_model_features, select_model_features_set

API_FEATURE_SET = "api_with_title"
KAGGLE_ARTIFACT_PATH = Path("models/model.joblib")
API_ARTIFACT_PATH = Path("models/api_model.joblib")


def select_training_features(df, profile):
    if profile == "api":
        return select_model_features_set(
            df,
            feature_set=API_FEATURE_SET,
            include_target=True,
        )

    return select_model_features(df, include_target=True)


def main(profile="kaggle"):
    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = load_processed_data()

    df = select_training_features(df, profile)

    features = [col for col in df.columns if col != "Survived"]

    # ---------------------------------------------------------
    # Train / test split
    # ---------------------------------------------------------

    X_train, y_train, _ = get_train_test_data(
        df,
        features=features,
        target="Survived",
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

    artifact = {
        "pipeline": pipeline,
        "model_name": model.__class__.__name__,
        "features": features,
        "shap_background": shap_background,
    }

    if profile == "kaggle":
        artifact.update(
            {
                "global_survival_rate": float(y_train.mean()),
                "fare_per_person_by_pclass": (
                    X_train.groupby("Pclass")["FarePerPerson_log1p"]
                    .median()
                    .to_dict()
                ),
                "global_fare_per_person_log1p": float(
                    X_train["FarePerPerson_log1p"].median()
                ),
            }
        )

    artifact_path = API_ARTIFACT_PATH if profile == "api" else KAGGLE_ARTIFACT_PATH
    artifact_path.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, artifact_path)

    print(f"{profile.capitalize()} model trained: {model.__class__.__name__}")
    print(f"Features used: {features}")
    print(f"Model saved to {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["api", "kaggle"], default="kaggle")
    args = parser.parse_args()
    main(profile=args.profile)
