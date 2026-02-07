import pandas as pd
import numpy as np
try:
    import statsmodels.api as SM
except ImportError:
    SM = None
except Exception: # Handle other potential import errors like the scipy one
    SM = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model using Statsmodels.
    Adds a constant/intercept automatically.
    """
    X_train_const = SM.add_constant(X_train)
    logit_model = SM.Logit(y_train, X_train_const).fit(disp=0)
    print("Logistic Regression model trained.")
    return logit_model

def get_logistic_predictions(model, X_test, threshold=0.5):
    """
    Gets predictions from statsmodels logit.
    """
    X_test_const = SM.add_constant(X_test)
    y_prob = model.predict(X_test_const)
    y_pred = (y_prob > threshold).astype(int)
    return y_pred, y_prob

def train_random_forest(X_train, y_train, random_state=42):
    """
    Trains a Random Forest Classifier with RandomizedSearchCV optimization.
    """
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 5, 10],
    }
    
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='recall',
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(f"Random Forest trained. Best params: {random_search.best_params_}")
    return best_model
