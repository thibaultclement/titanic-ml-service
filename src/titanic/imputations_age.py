import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def make_age_validation_split(df, target_col="Age", mask_frac=0.2, random_state=42):
    df_known = df[df[target_col].notna()].copy()

    rng = np.random.default_rng(random_state)
    masked_idx = rng.choice(
        df_known.index,
        size=int(len(df_known) * mask_frac),
        replace=False
    )

    df_masked = df_known.copy()
    y_true = df_masked.loc[masked_idx, target_col].copy()

    df_masked.loc[masked_idx, target_col] = np.nan

    return df_masked, masked_idx, y_true


def impute_age_global_median(df, age_col="Age"):
    df = df.copy()
    median_age = df[age_col].median()
    return df[age_col].fillna(median_age)


def impute_age_by_group_median(df, group_cols, age_col="Age"):
    df = df.copy()
    global_median = df[age_col].median()

    group_medians = (
        df
        .groupby(group_cols)[age_col]
        .transform("median")
    )

    return df[age_col].fillna(group_medians).fillna(global_median)


def impute_age_model(df, features, model, age_col="Age", categorical_features=None):
    df = df.copy()

    categorical_features = categorical_features or []
    numerical_features = [col for col in features if col not in categorical_features]

    train_mask = df[age_col].notna()
    pred_mask = df[age_col].isna()

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, age_col]
    X_pred = df.loc[pred_mask, features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    age_imputed = df[age_col].copy()

    pipeline.fit(X_train, y_train)

    if pred_mask.sum() > 0:
        age_imputed.loc[pred_mask] = pipeline.predict(X_pred)

    return age_imputed


def prepare_imputer_matrix(df, features, age_col="Age", categorical_features=None):
    df = df.copy()

    categorical_features = categorical_features or []
    numerical_features = [col for col in features if col not in categorical_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    X_features = preprocessor.fit_transform(df[features])

    age_values = df[[age_col]].to_numpy()

    matrix = np.hstack([age_values, X_features])

    return matrix


def impute_age_knn_imputer(
    df,
    features,
    categorical_features,
    age_col="Age",
    n_neighbors=3
):
    df = df.copy()

    matrix = prepare_imputer_matrix(
        df=df,
        features=features,
        age_col=age_col,
        categorical_features=categorical_features
    )

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights="distance"
    )

    imputed_matrix = imputer.fit_transform(matrix)

    age_imputed = pd.Series(
        imputed_matrix[:, 0],
        index=df.index,
        name=age_col
    )

    return age_imputed


def impute_age_iterative_extra_trees(
    df,
    features,
    categorical_features,
    age_col="Age",
    random_state=42,
    max_iter=50
):
    df = df.copy()

    matrix = prepare_imputer_matrix(
        df=df,
        features=features,
        age_col=age_col,
        categorical_features=categorical_features
    )

    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=100,
            random_state=random_state,
            min_samples_leaf=3,
            n_jobs=-1
        ),
        initial_strategy="median",
        max_iter=max_iter,
        random_state=random_state
    )

    imputed_matrix = imputer.fit_transform(matrix)

    age_imputed = pd.Series(
        imputed_matrix[:, 0],
        index=df.index,
        name=age_col
    )

    return age_imputed


def evaluate_age_imputation(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def benchmark_age_imputations(
    df,
    features,
    categorical_features,
    mask_frac=0.2,
    random_state=42
):
    df_masked, masked_idx, y_true = make_age_validation_split(
        df,
        target_col="Age",
        mask_frac=mask_frac,
        random_state=random_state
    )

    results = {}

    # 1. Médiane globale
    age_pred = impute_age_global_median(df_masked)
    results["Median"] = evaluate_age_imputation(
        y_true,
        age_pred.loc[masked_idx]
    )

    # 2. Médiane par Title
    if "Title" in df_masked.columns:
        age_pred = impute_age_by_group_median(df_masked, ["Title"])
        results["Median_by_Title"] = evaluate_age_imputation(
            y_true,
            age_pred.loc[masked_idx]
        )

    # 3. Médiane par Title + Pclass
    if all(col in df_masked.columns for col in ["Title", "Pclass"]):
        age_pred = impute_age_by_group_median(df_masked, ["Title", "Pclass"])
        results["Median_by_Title_Pclass"] = evaluate_age_imputation(
            y_true,
            age_pred.loc[masked_idx]
        )

    # 4. Régression linéaire
    age_pred = impute_age_model(
        df_masked,
        features=features,
        categorical_features=categorical_features,
        model=LinearRegression()
    )
    results["LinearRegression"] = evaluate_age_imputation(
        y_true,
        age_pred.loc[masked_idx]
    )

    # 5. KNN Regressor
    age_pred = impute_age_model(
        df_masked,
        features=features,
        categorical_features=categorical_features,
        model=KNeighborsRegressor(n_neighbors=5, weights="distance")
    )
    results["KNNRegressor"] = evaluate_age_imputation(
        y_true,
        age_pred.loc[masked_idx]
    )

    # 6. Random Forest
    age_pred = impute_age_model(
        df_masked,
        features=features,
        categorical_features=categorical_features,
        model=RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            min_samples_leaf=3,
            n_jobs=-1
        )
    )
    results["RandomForest"] = evaluate_age_imputation(
        y_true,
        age_pred.loc[masked_idx]
    )

    # 7. ExtraTreesRegressor
    age_pred = impute_age_model(
        df_masked,
        features=features,
        categorical_features=categorical_features,
        model=ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            min_samples_leaf=3,
            n_jobs=-1
        )
    )
    results["ExtraTreesRegressor"] = evaluate_age_imputation(
        y_true,
        age_pred.loc[masked_idx]
    )

    # KNNImputer
    age_pred = impute_age_knn_imputer(
        df_masked,
        features=features,
        categorical_features=categorical_features,
        n_neighbors=3
    )
    results["KNNImputer"] = evaluate_age_imputation(
        y_true,
        age_pred.loc[masked_idx]
    )

    # IterativeExtraTrees
    age_pred = impute_age_iterative_extra_trees(
        df_masked,
        features=features,
        categorical_features=categorical_features,
        random_state=random_state,
        max_iter=50
    )
    results["IterativeExtraTrees"] = evaluate_age_imputation(
        y_true,
        age_pred.loc[masked_idx]
    )

    results_df = (
        pd.DataFrame(results)
        .T
        .sort_values("MAE")
    )

    return results_df