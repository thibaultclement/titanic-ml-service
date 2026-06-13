# scripts/predict.py

from pathlib import Path
import joblib

from titanic.data import load_processed_data
from titanic.selection import select_model_features
from titanic.predict import make_predictions, save_submission


def main():
    model_bundle = joblib.load("models/model.joblib")

    pipeline = model_bundle["pipeline"]
    features = model_bundle["features"]

    df = load_processed_data()

    df_model = select_model_features(
        df,
        include_target=True
    )

    test_df = df_model[df_model["Survived"].isna()].copy()
    X_test = test_df[features]

    predictions_df = make_predictions(
        model=pipeline,
        X_test=X_test,
        passenger_ids=test_df.index
    )

    Path("data/submissions").mkdir(parents=True, exist_ok=True)

    save_submission(
        predictions_df,
        filepath="data/submissions/submission.csv"
    )

    print("Submission saved successfully")
    print(predictions_df.shape)
    print(predictions_df.head())


if __name__ == "__main__":
    main()