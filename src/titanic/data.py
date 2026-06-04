# src/titanic/data.py

from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
INTERIM_DATA_DIR = Path("data/interim")
PROCESSED_DATA_DIR = Path("data/processed")


def load_train(path=RAW_DATA_DIR / "train.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_test(path=RAW_DATA_DIR / "test.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_raw_data(
    train_path=RAW_DATA_DIR / "train.csv",
    test_path=RAW_DATA_DIR / "test.csv",
) -> pd.DataFrame:
    train = load_train(train_path)
    test = load_test(test_path)

    df = pd.concat([train, test], axis=0, ignore_index=True)

    return df


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_interim_data(path=INTERIM_DATA_DIR / "titanic.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_processed_data(path=PROCESSED_DATA_DIR / "features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)