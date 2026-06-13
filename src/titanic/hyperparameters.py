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
            "n_estimators": [25, 50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
    },

    "SVC": {
        "model": SVC(),
        "params": {
            "C": [0.1, 0.5, 1, 2, 5, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
    },

    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    },

    "LogisticRegression": {
        "model": LogisticRegression(max_iter=2000),
        "params": {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        },
    },

    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [1, 2, 3],
            "subsample": [0.8, 1.0],
        },
    },

    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 8, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        },
    },

    "XGBClassifier": {
        "model": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
        ),
        "params": {
            "max_depth": [2, 3, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 300],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    },
}