# scripts/build_dataset.py

from titanic.data import load_raw_data, save_dataset
from titanic.cleaning import clean_data
from titanic.features import build_features
from titanic.survival_features import add_survival_rate_features


def build_dataset():
    df_raw = load_raw_data()

    save_dataset(
        df_raw,
        "data/interim/titanic_merged.parquet"
    )

    df_cleaned = clean_data(df_raw)

    save_dataset(
        df_cleaned,
        "data/processed/titanic_cleaned.parquet"
    )

    df_features = build_features(df_cleaned)
    df_features = add_survival_rate_features(df_features)


    save_dataset(
        df_features,
        "data/processed/titanic_features.parquet"
    )

    print("Dataset built successfully")
    print(f"Raw shape: {df_raw.shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Features shape: {df_features.shape}")


if __name__ == "__main__":
    build_dataset()