import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df):
    """Encodes categorical fields, scales numeric fields, and splits data into train/test sets."""
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    
    # Encode binary categorical columns (0/1)
    for col in binary_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
    # One-hot encode multi-class categorical columns
    X = pd.get_dummies(X, columns=multi_cols, dtype=int)
    
    # Scale numeric columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    feature_names = list(X.columns)
    
    # Split dataset into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    return X_train, X_test, y_train, y_test, feature_names, scaler

def preprocess_single(input_dict, feature_names, scaler):
    """Preprocesses a single user input from the Streamlit UI to match model expectations."""
    # Initialize empty row with expected training feature names
    single_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Assign scaled variables
    for col in num_cols:
        if col in input_dict:
            single_df.loc[0, col] = float(input_dict[col])
            
    # Assign binary fields
    binary_fields = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_fields:
        if col in input_dict:
            single_df.loc[0, col] = float(binary_map.get(input_dict[col], 0))
            
    if 'SeniorCitizen' in input_dict:
        single_df.loc[0, 'SeniorCitizen'] = float(binary_map.get(input_dict['SeniorCitizen'], 0))
        
    # Process multi-class categorical columns as one-hot fields
    multi_fields = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    
    for col in multi_fields:
        if col in input_dict:
            val = input_dict[col]
            dummy_col = f"{col}_{val}"
            if dummy_col in feature_names:
                single_df.loc[0, dummy_col] = 1.0
        else:
            dummy_col = f"{col}_No"
            if dummy_col in feature_names:
                single_df.loc[0, dummy_col] = 1.0
                
    # Normalize numeric fields using the fitted training scaler
    single_df[num_cols] = scaler.transform(single_df[num_cols])
    
    return single_df
