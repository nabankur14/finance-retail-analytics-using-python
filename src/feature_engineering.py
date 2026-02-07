import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    """
    Calculate Variance Inflation Factor (VIF) for each variable in a DataFrame.
    """
    vif_df = pd.DataFrame()
    vif_df["Variable"] = df.columns
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_df

def drop_high_vif_features(X_train, X_test, threshold=5):
    """
    Iteratively drops features with VIF greater than the threshold.
    """
    df = X_train.copy()
    high_vif = True
    
    while high_vif:
        vif_df = calculate_vif(df)
        max_vif = vif_df['VIF'].max()
        
        if max_vif > threshold:
            feature_to_drop = vif_df.sort_values('VIF', ascending=False).iloc[0]['Variable']
            print(f"Dropping {feature_to_drop} with VIF: {max_vif:.2f}")
            df = df.drop(columns=[feature_to_drop])
        else:
            high_vif = False
            
    cols_to_keep = df.columns
    print(f"Final feature set selected. retained {len(cols_to_keep)} features.")
    return X_train[cols_to_keep], X_test[cols_to_keep]
