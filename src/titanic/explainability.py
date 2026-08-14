# src/titanic/explainability.py

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix


def load_model_bundle(path="models/model.joblib"):
    return joblib.load(path)


def get_train_test_from_model_df(df_model, target="Survived"):
    train_df = df_model[df_model[target].notna()].copy()
    test_df = df_model[df_model[target].isna()].copy()

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)

    X_test = test_df.drop(columns=[target])

    return X_train, y_train, X_test


def compute_permutation_importance(
    pipeline,
    X,
    y,
    scoring="accuracy",
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
):
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def get_prediction_errors(pipeline, X, y):
    y_pred = pipeline.predict(X)

    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[:, 1]

    errors = X.copy()
    errors["y_true"] = y.values
    errors["y_pred"] = y_pred
    errors["is_error"] = errors["y_true"] != errors["y_pred"]

    if proba is not None:
        errors["survival_probability"] = proba

    return errors


def print_classification_summary(pipeline, X, y):
    y_pred = pipeline.predict(X)

    print("Classification report:")
    print(classification_report(y, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))


def compute_shap_kernel_explainer(pipeline, X_background):
    import shap

    def predict_proba_fn(X):
        X = pd.DataFrame(X, columns=X_background.columns)
        return pipeline.predict_proba(X)[:, 1]

    explainer = shap.KernelExplainer(
        predict_proba_fn, shap.sample(X_background, min(100, len(X_background)))
    )

    return explainer


def compute_shap_values(explainer, X_sample, nsamples=100):
    return explainer.shap_values(X_sample, nsamples=nsamples)
