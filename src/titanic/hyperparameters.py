from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


MODEL_GRIDS = {
    "AdaBoostClassifier": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "model__n_estimators": [25, 50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
    },

    "SVC": {
        "model": SVC(),
        "params": {
            "model__C": [0.1, 0.5, 1, 2, 5, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        },
    },

    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    },

    "LogisticRegression": {
        "model": LogisticRegression(max_iter=2000),
        "params": {
            "model__C": [0.001, 0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
        },
    },

    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [1, 2, 3],
            "model__subsample": [0.8, 1.0],
        },
    },

    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": [None, "balanced"],
        },
    },

    "XGBClassifier": {
        "model": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
        ),
        "params": {
            "model__max_depth": [2, 3, 5],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__n_estimators": [100, 300],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    },
}