import pandas as pd

def load_data(filepath):
    """Loads Telco Customer Churn dataset, handles missing charges, and encodes the target variable."""
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nDataset Head:\n", df.head())
    
    # Handle TotalCharges (convert strings to numeric, fill missing with median)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop(columns=['customerID'])
    total_charges_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print("\nClass Distribution of Churn:\n", df['Churn'].value_counts())
    
    return df
