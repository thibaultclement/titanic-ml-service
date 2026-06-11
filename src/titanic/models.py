from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier


def get_baseline_models(random_state=42):
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVC": SVC(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
        "RandomForestClassifier": RandomForestClassifier(random_state=random_state),
        "ExtraTreesClassifier": ExtraTreesClassifier(random_state=random_state),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=random_state),
        "BaggingClassifier": BaggingClassifier(random_state=random_state),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
        "GaussianNB": GaussianNB(),
        "XGBClassifier": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        ),
    }