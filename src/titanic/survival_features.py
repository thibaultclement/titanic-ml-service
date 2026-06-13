# src/titanic/survival_features.py

import pandas as pd


def add_group_survival_rate(
    df,
    group_col,
    target="Survived",
    feature_name=None,
    alpha=5,
):
    """
    Ajoute une feature de survie moyenne par groupe.

    Pour les lignes train :
    - utilise un leave-one-out pour éviter que la ligne voie sa propre cible.

    Pour les lignes test :
    - utilise le taux de survie observé dans le train pour le même groupe.

    alpha applique un lissage vers la moyenne globale.
    """
    df = df.copy()

    if feature_name is None:
        feature_name = f"{group_col}SurvivalRate"

    train_mask = df[target].notna()
    y_train = df.loc[train_mask, target]

    global_rate = y_train.mean()

    stats = (
        df.loc[train_mask]
        .groupby(group_col)[target]
        .agg(["sum", "count"])
    )

    group_sum = df[group_col].map(stats["sum"]).fillna(0)
    group_count = df[group_col].map(stats["count"]).fillna(0)

    # Test rows : taux du groupe observé dans le train
    rate = (group_sum + alpha * global_rate) / (group_count + alpha)

    # Train rows : leave-one-out pour ne pas utiliser sa propre valeur Survived
    loo_sum = group_sum.loc[train_mask] - y_train
    loo_count = group_count.loc[train_mask] - 1

    loo_rate = (loo_sum + alpha * global_rate) / (loo_count + alpha)

    rate.loc[train_mask] = loo_rate

    df[feature_name] = rate
    df[f"{feature_name}Count"] = group_count

    return df


def add_survival_rate_features(df):
    df = df.copy()

    df = add_group_survival_rate(
        df,
        group_col="Surname",
        feature_name="FamilySurvivalRate",
        alpha=5,
    )

    df = add_group_survival_rate(
        df,
        group_col="Ticket",
        feature_name="TicketSurvivalRate",
        alpha=5,
    )

    return df