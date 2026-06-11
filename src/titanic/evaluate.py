# evaluate.py

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_responses(path="../data/responses/responses.csv"):
    return pd.read_csv(
        path,
        index_col="PassengerId",
        usecols=["PassengerId", "Survived"]
    )


def evaluate_submission(
    predictions_df,
    responses_path="../data/responses/responses.csv",
    verbose=True
):
    responses = load_responses(responses_path)

    submission = predictions_df.copy()

    if "Survived" not in submission.columns:
        raise ValueError("predictions_df must contain a 'Survived' column.")

    common_index = submission.index.intersection(responses.index)

    if len(common_index) != len(responses):
        raise ValueError(
            f"Index mismatch: compared {len(common_index)} rows, "
            f"but responses contains {len(responses)} rows."
        )

    y_pred = submission.loc[common_index, "Survived"].astype(int)
    y_true = responses.loc[common_index, "Survived"].astype(int)

    score = accuracy_score(y_true, y_pred)

    if verbose:
        print(f"Local Kaggle Score: {score:.4f}")
        print(f"Compared rows: {len(common_index)} / {len(responses)}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_true, y_pred))

    return score