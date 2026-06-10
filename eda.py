import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    """Generates and saves exploratory data analysis plots to the plots/ folder."""
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Churn Distribution
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Churn', data=df, hue='Churn', palette='viridis', legend=False)
    plt.title('Distribution of Customer Churn', fontsize=14, pad=15)
    plt.xlabel('Churn Status (0 = Retained, 1 = Churned)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/churn_distribution.png', dpi=150)
    plt.close()
    
    # 2. Monthly Charges Density by Churn
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='crest', alpha=0.5, linewidth=2)
    plt.title('Monthly Charges Distribution by Churn Status', fontsize=14, pad=15)
    plt.xlabel('Monthly Charges ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Churn', labels=['Churned (1)', 'Retained (0)'])
    plt.tight_layout()
    plt.savefig('plots/monthly_charges_by_churn.png', dpi=150)
    plt.close()
    
    # 3. Tenure vs Churn
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='Churn', y='tenure', data=df, hue='Churn', palette='coolwarm', legend=False)
    plt.title('Customer Tenure by Churn Status', fontsize=14, pad=15)
    plt.xlabel('Churn Status (0 = Retained, 1 = Churned)', fontsize=12)
    plt.ylabel('Tenure (Months)', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/tenure_by_churn.png', dpi=150)
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numerical_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_cols.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numerical Features', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=150)
    plt.close()
    
    print("EDA complete. Plots saved in plots/ folder.")
