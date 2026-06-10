import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

def evaluate_models(models, X_test, y_test):
    """Evaluates classifier performance and generates confusion matrices and ROC curves."""
    os.makedirs('plots', exist_ok=True)
    results = {}
    
    display_names = {
        'LogisticRegression': 'logistic_regression',
        'RandomForest': 'random_forest',
        'XGBoost': 'xgboost'
    }
    
    roc_curves_data = {}
    
    for name, model in models.items():
        print(f"\n================ Evaluating {name} ================")
        
        y_pred = model.predict(X_test)
        
        # Extract prediction probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"Accuracy:        {acc:.4f}")
        print(f"F1 Score (W):    {f1:.4f}")
        print(f"ROC-AUC:         {auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        results[name] = {
            'accuracy': acc,
            'f1': f1,
            'roc_auc': auc
        }
        
        # Calculate ROC data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[name] = (fpr, tpr, auc)
        
        # Confusion Matrix Plot
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Retained (0)', 'Churned (1)'],
                    yticklabels=['Retained (0)', 'Churned (1)'])
        plt.title(f'{name} Confusion Matrix', fontsize=14, pad=15)
        plt.ylabel('Actual Status', fontsize=12)
        plt.xlabel('Predicted Status', fontsize=12)
        plt.tight_layout()
        
        file_name = display_names.get(name, name.lower())
        plt.savefig(f'plots/confusion_matrix_{file_name}.png', dpi=150)
        plt.close()
        
    # Grouped Bar Chart Comparison
    print("\nSaving model comparison bar chart...")
    comparison_list = []
    for m_name, metrics in results.items():
        comparison_list.append({'Model': m_name, 'Metric': 'Accuracy', 'Value': metrics['accuracy']})
        comparison_list.append({'Model': m_name, 'Metric': 'F1 Score', 'Value': metrics['f1']})
        comparison_list.append({'Model': m_name, 'Metric': 'ROC-AUC', 'Value': metrics['roc_auc']})
        
    df_compare = pd.DataFrame(comparison_list)
    
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=df_compare, x='Model', y='Value', hue='Metric', palette='viridis')
    plt.title('Model Performance Comparison', fontsize=14, pad=15)
    plt.ylabel('Score Value', fontsize=12)
    plt.xlabel('Classifier', fontsize=12)
    plt.ylim(0, 1.1)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
            
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=150)
    plt.close()
    
    # Combined ROC curves
    print("Saving combined ROC curves...")
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (0.50)')
    
    colors = {'LogisticRegression': 'blue', 'RandomForest': 'green', 'XGBoost': 'red'}
    for m_name, (fpr, tpr, auc_score) in roc_curves_data.items():
        color = colors.get(m_name, 'black')
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{m_name} (AUC = {auc_score:.2f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png', dpi=150)
    plt.close()
    
    return results
