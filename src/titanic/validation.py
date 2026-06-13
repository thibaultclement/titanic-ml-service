import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from titanic.models import build_model_pipeline


def benchmark_models(
    models,
    preprocessor,
    X,
    y,
    scoring="accuracy",
    n_splits=10,
    random_state=42,
):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    results = []

    for name, model in models.items():
        pipeline = build_model_pipeline(
            preprocessor=preprocessor,
            model=model
        )

        cv_results = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            error_score="raise"
        )

        results.append({
            "model": name,
            "train_score_mean": cv_results["train_score"].mean(),
            "train_score_std": cv_results["train_score"].std(),
            "test_score_mean": cv_results["test_score"].mean(),
            "test_score_std": cv_results["test_score"].std(),
        })

    return (
        pd.DataFrame(results)
        .sort_values("test_score_mean", ascending=False)
        .reset_index(drop=True)
    )