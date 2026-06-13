from sklearn.model_selection import GridSearchCV, StratifiedKFold
from titanic.models import build_model_pipeline


def tune_model(
    model,
    param_grid,
    preprocessor,
    X,
    y,
    scoring="accuracy",
    n_splits=5,
    random_state=42,
    n_jobs=-1,
    verbose=0,
):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    pipeline = build_model_pipeline(
        preprocessor=preprocessor,
        model=model
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise"
    )

    grid.fit(X, y)

    return grid