import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X_train, y_train):
    """Trains Logistic Regression, Random Forest, and XGBoost classifiers, then saves them to disk."""
    os.makedirs('models', exist_ok=True)
    models = {}
    
    # Model 1: Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    models['LogisticRegression'] = lr_model
    
    # Model 2: Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/random_forest.pkl')
    models['RandomForest'] = rf_model
    
    # Model 3: XGBoost Classifier
    print("Training XGBoost...")
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, 'models/xgboost.pkl')
    models['XGBoost'] = xgb_model
    
    print("Models saved in models/ folder.")
    return models
