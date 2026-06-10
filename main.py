import os
import joblib
import pandas as pd

from data_loader import load_data
from eda import run_eda
from preprocess import preprocess
from train import train_models
from evaluate import evaluate_models
from explain import shap_explain

if __name__ == '__main__':
    dataset_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please place it in the same root folder.")
        
    print("="*50)
    print("STEP 1: Loading and cleaning dataset...")
    print("="*50)
    df = load_data(dataset_path)
    
    print("\n" + "="*50)
    print("STEP 2: Running Exploratory Data Analysis (EDA)...")
    print("="*50)
    run_eda(df)
    
    print("\n" + "="*50)
    print("STEP 3: Preprocessing data and creating splits...")
    print("="*50)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)
    
    # Save the fitted scaler and feature names for web app usage
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("Scaler and feature names successfully saved in models/ folder.")
    
    print("\n" + "="*50)
    print("STEP 4: Training machine learning models...")
    print("="*50)
    models = train_models(X_train, y_train)
    
    print("\n" + "="*50)
    print("STEP 5: Evaluating models and generating metric plots...")
    print("="*50)
    results = evaluate_models(models, X_test, y_test)
    
    print("\n" + "="*50)
    print("STEP 6: Calculating model explainability with SHAP...")
    print("="*50)
    shap_explain(models['XGBoost'], X_test)
    
    print("\n" + "="*50)
    print("STEP 7: Printing final performance comparison summary table...")
    print("="*50)
    df_summary = pd.DataFrame(results).T
    df_summary.index.name = 'Model'
    print(df_summary.round(4))
    print("="*50)
    print("\nML pipeline execution complete! Models and plots are saved.")
    print("Now you can start the Streamlit web app using: streamlit run app.py")
