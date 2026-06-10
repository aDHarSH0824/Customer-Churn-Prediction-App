import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

def shap_explain(xgb_model, X_test):
    """Calculates SHAP values for the XGBoost model and logs the top 5 most important features."""
    os.makedirs('plots', exist_ok=True)
    
    print("\nCalculating SHAP values for XGBoost model...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    # Normalize shap_values return format (handles single array or list of arrays)
    if isinstance(shap_values, list):
        shap_values_to_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_to_use = shap_values
        
    # Save global SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_use, X_test, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', dpi=150)
    plt.close()
    
    # Calculate top features by mean absolute SHAP value
    if len(shap_values_to_use.shape) == 2:
        mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)
    elif len(shap_values_to_use.shape) == 3:
        mean_abs_shap = np.abs(shap_values_to_use).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)
        
    importances = pd.Series(mean_abs_shap, index=X_test.columns)
    top_5 = importances.sort_values(ascending=False).head(5)
    
    print("\nTop 5 most important features based on SHAP values:")
    for rank, (feature, val) in enumerate(top_5.items(), 1):
        print(f"{rank}. {feature} (Mean |SHAP| = {val:.4f})")
        
    print("\nSHAP explanation complete.")
