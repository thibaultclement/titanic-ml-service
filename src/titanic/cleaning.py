# titanic.cleaning

import numpy as np
import pandas as pd

from collections import Counter


# ---------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------

def detect_outliers_tukey(df, features, n=0, iqr_multiplier=1.5):
    outlier_indices = []

    for col in features:
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col], 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        outlier_list_col = df[
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ].index

        outlier_indices.extend(outlier_list_col)

    outlier_counts = Counter(outlier_indices)

    multiple_outliers = [
        index for index, count in outlier_counts.items()
        if count > n
    ]

    return multiple_outliers


def set_outliers_to_nan(df, var, threshold):
    df = df.copy()
    df.loc[df[var] >= threshold, var] = np.nan
    return df


# ---------------------------------------------------------------------
# Generic imputers
# ---------------------------------------------------------------------

def fill_with_mode(df, var):
    return df[var].fillna(df[var].mode()[0])


def fill_with_value(df, var, value):
    return df[var].fillna(value)


def fill_with_group_median(df, var, group_cols):
    group_median = df.groupby(group_cols)[var].transform("median")
    global_median = df[var].median()

    return df[var].fillna(group_median).fillna(global_median)


# ---------------------------------------------------------------------
# Titanic-specific cleaning rules
# ---------------------------------------------------------------------

def impute_fare(df):
    """
    Impute missing Fare values using the median Fare
    of passengers with the same Pclass and Embarked.

    Fallback: median by Pclass.
    Final fallback: global median.
    """
    df = df.copy()

    missing_mask = df["Fare"].isna()

    fare_by_pclass_embarked = (
        df
        .groupby(["Pclass", "Embarked"])["Fare"]
        .transform("median")
    )

    fare_by_pclass = (
        df
        .groupby("Pclass")["Fare"]
        .transform("median")
    )

    global_fare_median = df["Fare"].median()

    df["Fare"] = (
        df["Fare"]
        .fillna(fare_by_pclass_embarked)
        .fillna(fare_by_pclass)
        .fillna(global_fare_median)
    )

    return df


def impute_embarked(df, value="C"):
    """
    Impute missing Embarked values.

    In the Titanic dataset, the two missing Embarked values
    correspond to first-class passengers with a Fare around 80,
    which is more consistent with Cherbourg ('C') than with the
    global mode ('S').
    """
    df = df.copy()
    df["Embarked"] = df["Embarked"].fillna(value)
    return df


def clean_data(df):
    """
    Apply simple deterministic cleaning steps.

    Age imputation is intentionally excluded because it is handled
    in imputations_age.py as a feature engineering / modeling choice.
    """
    df = df.copy()

    df = impute_embarked(df, value="C")
    df = impute_fare(df)

    return df