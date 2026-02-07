import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Loads data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_column_names(df):
    """
    Cleans column names by removing special characters and replacing spaces with underscores.
    """
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('%', 'perc').str.replace('/', '_to_')
    print("Column names cleaned.")
    return df

def handle_missing_values(df, n_neighbors=10):
    """
    Imputes missing values using KNNImputer.
    Assumes df contains only numerical columns for imputation or handled externally.
    """
    # Select numeric columns for imputation
    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric_df = df.select_dtypes(exclude=[np.number])

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(numeric_df)
    
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)
    
    # Combine back
    final_df = pd.concat([imputed_df, non_numeric_df], axis=1)
    print("Missing values imputed using KNN.")
    return final_df

def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    print("Features scaled.")
    return X_train_scaled, X_test_scaled, scaler
