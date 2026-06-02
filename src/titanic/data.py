# src/titanic/data.py

import pandas as pd

def prepare_dataset(df: pd.DataFrame, target: str = "Survived"):
    df = df.copy()

    y = df[target] if target in df.columns else None
    X = df.drop(columns=[target], errors="ignore")

    X = clean_data(X)
    X = build_features(X)
    X = select_model_features(X)

    return X, y