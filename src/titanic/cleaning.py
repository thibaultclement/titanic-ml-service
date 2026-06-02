# titanic.cleaning

import pandas as pd
import numpy as np
import re


def outliers_detect(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    
    from collections import Counter
    
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        
        print(col)
        
        # 1st quartile (25%)
        Q1 = np.nanpercentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.nanpercentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers

def outliers_delete(df, var, value):
    df.loc[df[var]>=value,var] = np.nan
    return df[var]

def age_median(df):
    mediane = df["Age"].fillna(value=df["Age"].median())
    return mediane

def age_median_by_title(df):
    df_copy = df.copy()
    df_copy["AgeMedianTitle"] = df_copy["Age"]
    median_age_by_title = df_copy.groupby('Title')['Age'].median()
    for title, median_age in median_age_by_title.items():
        df_copy.loc[(df_copy['Age'].isnull()) & (df_copy['Title'] == title), 'AgeMedianTitle'] = median_age
    return df_copy["AgeMedianTitle"]

def imputer_knn(df, var):
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import KNNImputer
    
    original_index = df.index
    df_reset_index = df.reset_index(drop=True)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_reset_index)
    knn_imputer = KNNImputer(n_neighbors=3, weights='distance')
    df_imputed = knn_imputer.fit_transform(df_scaled)
    df_imputed = pd.DataFrame(
        columns=df.columns,
        data=scaler.inverse_transform(df_imputed)
    )
    df_imputed.index = original_index
    return df_imputed[var]


def age_etr(df):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import IterativeImputer
    
    df_reset_index = df.reset_index(drop=True)

    miss_forest = IterativeImputer(
        estimator=ExtraTreesRegressor(),
        initial_strategy='median',
        max_iter=200
    )
    tree_imputed = miss_forest.fit_transform(df_reset_index)
    tree_imputed_df = pd.DataFrame(data=tree_imputed,columns=df_reset_index.columns)
    
    tree_imputed_df.index = df.index
    
    return tree_imputed_df["Age"]

def imputer_mode(df, var):
    return df[var].fillna(df[var].mode()[0])

def imputer_X(df, var, imput):
    return df[var].fillna(imput)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Embarked"] = fill_with_mode(df, "Embarked")
    df["Fare"] = fill_with_median(df, "Fare")
    return df