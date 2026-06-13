from sklearn.model_selection import GridSearchCV, StratifiedKFold


def tune_model(
    model,
    param_grid,
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

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise"
    )

    grid.fit(X, y)

    return grid