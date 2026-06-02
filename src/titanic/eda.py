# eda

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as spstats

def stats(df):

    stats = df.describe(include='all').T

    # Calcul du nombre réel de valeurs uniques pour chaque variable numérique
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):

            # unique
            unique_count = df[col].nunique()
            stats.at[col, 'unique'] = unique_count

            # top
            top_value = df[col].mode()[0]
            stats.at[col, 'top'] = top_value

            # freq
            freq_value = df[col].value_counts()[top_value]
            stats.at[col, 'freq'] = freq_value

    stats[['count', 'unique', 'freq']] = stats[['count', 'unique', 'freq']].astype(int)
    stats = stats.fillna("/")
    stats.insert(0, 'type', df.dtypes)

    total_values = df.shape[0]
    null_percentage = (df.isnull().sum() / total_values) * 100

    null_counts = df.isnull().sum()
    null_percentage_str = (np.ceil(null_percentage)).astype(int).astype(str) + '%'
    stats.insert(stats.columns.get_loc('count') + 1, 'null', null_counts.astype(str) + ' (' + null_percentage_str + ')')

    return stats


def describe_numeric(df, col):

    display(df[col].describe().to_frame().T)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.histplot(
        df[col].dropna(),
        kde=True,
        ax=axes[0]
    )
    axes[0].set_title(f"Distribution de {col}")

    sns.boxplot(
        x=df[col].dropna(),
        ax=axes[1]
    )
    axes[1].set_title(f"Boxplot de {col}")

    spstats.probplot(
        df[col].dropna(),
        dist="norm",
        plot=axes[2]
    )
    axes[2].set_title(f"QQ-plot de {col}")

    plt.tight_layout()
    plt.show()

    print(f"Missing values : {df[col].isna().sum()}")
    print(f"Skewness       : {df[col].skew():.3f}")
    print(f"Kurtosis       : {df[col].kurtosis():.3f}")


def benchmark_age_figure(df):
    plt.figure(figsize=(10, 6))

    df["Age"].plot.kde(color='orange')
    df["AgeMedian"].plot.kde(color='blue')
    df["AgeMedianTitle"].plot.kde(color='green')
    df["AgeKNN"].plot.kde(color='red')
    df["AgeETR"].plot.kde(color='purple')

    plt.title('Density distribution of the different variables for filling in the null values of Age')
    plt.legend()
    plt.show()


def benchmark_age_table(df):
    StdAge = round(df["Age"].std(), 2)
    StdAgeMedian = round(df["AgeMedian"].std(), 2)
    StdAgeMedianTitle = round(df["AgeMedianTitle"].std(), 2)
    StdAgeKNN = round(df["AgeKNN"].std(), 2)
    StdAgeETR = round(df["AgeETR"].std(), 2)

    std_df = pd.DataFrame({'Std': [
        StdAge,
        StdAgeMedian,
        StdAgeMedianTitle,
        StdAgeKNN,
        StdAgeETR
    ]}, index=[
        'Age',
        'AgeMedian',
        'AgeMedianTitle',
        'AgeKNN',
        'AgeETR'
    ])

    return std_df