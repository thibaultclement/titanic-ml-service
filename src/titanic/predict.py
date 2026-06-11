import pandas as pd


def make_predictions(model, X_test, passenger_ids=None):
    predictions = model.predict(X_test).astype(int)

    if passenger_ids is None:
        index = X_test.index
    else:
        index = passenger_ids

    return pd.DataFrame(
        predictions,
        index=index,
        columns=["Survived"]
    )


def save_submission(predictions_df, filepath):
    predictions_df.to_csv(filepath, index=True)
    return filepath


def load_submission(filepath):
    return pd.read_csv(filepath, index_col="PassengerId")