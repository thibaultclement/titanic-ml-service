# scripts/evaluate.py

from titanic.evaluate import evaluate_submission
from titanic.predict import load_submission


def main():
    predictions_df = load_submission(
        "data/submissions/submission.csv"
    )

    evaluate_submission(
        predictions_df,
        responses_path="data/responses/responses.csv"
    )


if __name__ == "__main__":
    main()