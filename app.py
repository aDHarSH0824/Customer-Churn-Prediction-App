import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import textwrap

from preprocess import preprocess_single

# Page configurations
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark-theme styling and alerts
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
            color: #fafafa;
        }
        
        .header-container {
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        .header-title {
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .header-subtitle {
            color: #a1a1aa;
            font-size: 1.2rem;
            font-weight: 300;
        }
        
        [data-testid="stSidebar"] {
            background-color: #111318 !important;
            color: #fafafa !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        [data-testid="stSidebar"] [class*="stMarkdown"] p, 
        [data-testid="stSidebar"] [class*="stMarkdown"] h1, 
        [data-testid="stSidebar"] [class*="stMarkdown"] h2, 
        [data-testid="stSidebar"] [class*="stMarkdown"] h3, 
        [data-testid="stSidebar"] [class*="stMarkdown"] h4, 
        [data-testid="stSidebar"] [class*="stMarkdown"] h5, 
        [data-testid="stSidebar"] [class*="stMarkdown"] h6 {
            color: #fafafa !important;
        }
        
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 15px;
        }
        
        div.stButton > button:first-child:hover {
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
            transform: translateY(-2px);
            color: white !important;
        }
        
        .result-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 28px;
            margin-top: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        .alert-box {
            display: flex;
            align-items: center;
            border-radius: 10px;
            padding: 16px;
            margin-top: 15px;
            border: 1px solid transparent;
        }
        
        .alert-success {
            background-color: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }
        
        .alert-danger {
            background-color: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }
        
        .risk-bar-bg {
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            height: 20px;
            width: 100%;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .risk-bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("""
    <div class="header-container">
        <div class="header-title">Customer Churn Prediction App</div>
        <div class="header-subtitle">Enter customer details to predict if they will churn in real-time</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Info Panel
st.sidebar.markdown("# About This Project")
st.sidebar.markdown(
    "Predicts customer churn in the telecommunications sector using "
    "supervised Machine Learning. Understanding churn lets companies "
    "proactively target high-risk customers with retention offers."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Technical Overview")
st.sidebar.markdown("**Model Used:** XGBoost Classifier")
st.sidebar.markdown("**Dataset:** Telco Customer Churn (Kaggle)")
st.sidebar.markdown("**Accuracy:** ~92% | F1 Score: ~88%")
st.sidebar.markdown("*(Reflecting initial dataset splits)*")

# User Input Form Sections
st.markdown("### 📋 Enter Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 👤 Customer Profile")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select customer's gender.")
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"], help="Is the customer 65 years or older?")
    Partner = st.selectbox("Partner", ["No", "Yes"], help="Does the customer have a partner?")
    Dependents = st.selectbox("Dependents", ["No", "Yes"], help="Does the customer have dependents?")
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, help="Number of months the customer has stayed with the company.")

with col2:
    st.markdown("##### 🌐 Services Subscribed")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"], help="Does the customer have a phone line?")
    
    # Conditional logic for Phone Service features
    has_phone = PhoneService == "Yes"
    phone_opts = ["No", "Yes", "No phone service"] if has_phone else ["No phone service"]
    MultipleLines = st.selectbox("Multiple Lines", phone_opts, index=0 if has_phone else 0, help="Does the customer have multiple phone lines?")
    
    InternetService = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"], help="Type of internet service provider.")
    
    # Conditional logic for Internet Service options
    has_internet = InternetService != "No"
    sec_opts = ["No", "Yes", "No internet service"] if has_internet else ["No internet service"]
    
    OnlineSecurity = st.selectbox("Online Security", sec_opts, index=0 if has_internet else 0, help="Does the customer have online security add-on?")
    OnlineBackup = st.selectbox("Online Backup", sec_opts, index=0 if has_internet else 0, help="Does the customer have online backup add-on?")
    DeviceProtection = st.selectbox("Device Protection", sec_opts, index=0 if has_internet else 0, help="Does the customer have device protection add-on?")
    TechSupport = st.selectbox("Tech Support", sec_opts, index=0 if has_internet else 0, help="Does the customer have technical support add-on?")
    StreamingTV = st.selectbox("Streaming TV", sec_opts, index=0 if has_internet else 0, help="Does the customer use streaming TV service?")
    StreamingMovies = st.selectbox("Streaming Movies", sec_opts, index=0 if has_internet else 0, help="Does the customer use streaming movies service?")

with col3:
    st.markdown("##### 💳 Billing & Contract")
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], help="The contract term of the customer.")
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"], help="Whether the customer uses paperless billing.")
    PaymentMethod = st.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        help="How the customer pays their bills."
    )
    MonthlyCharges = st.slider("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=70.0, step=0.5, help="The amount charged to the customer monthly.")
    TotalCharges = st.slider("Total Charges ($)", min_value=0.0, max_value=9000.0, value=840.0, step=10.0, help="The total amount charged to the customer so far.")

# Assemble input data
input_dict = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

predict_clicked = st.button("Predict Churn")

if predict_clicked:
    if not (os.path.exists("models/xgboost.pkl") and os.path.exists("models/scaler.pkl") and os.path.exists("models/feature_names.pkl")):
        st.error("⚠️ Error: Model files not found. Please run 'python main.py' in the terminal first to train and generate the files.")
    else:
        xgb_model = joblib.load("models/xgboost.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        
        # Format input and predict
        processed_input = preprocess_single(input_dict, feature_names, scaler)
        prediction = xgb_model.predict(processed_input)[0]
        probability = xgb_model.predict_proba(processed_input)[0, 1]
        
        # Color coding metrics based on severity of risk
        if probability < 0.3:
            risk_color = "#34d399"
            bar_gradient = "linear-gradient(90deg, #10b981 0%, #059669 100%)"
            alert_html = f"""<div class="alert-box alert-success">
    <span style="font-size: 1.5rem; margin-right: 8px;">✅</span>
    <div>
        <strong>This customer is NOT likely to churn</strong><br/>
        <span style="font-size: 0.95rem;">Retention Probability: <strong>{(1 - probability) * 100:.2f}%</strong></span>
    </div>
</div>"""
        elif probability < 0.7:
            risk_color = "#fbbf24"
            bar_gradient = "linear-gradient(90deg, #fbbf24 0%, #d97706 100%)"
            alert_html = f"""<div class="alert-box alert-danger" style="background-color: rgba(251, 191, 36, 0.1); border-color: rgba(251, 191, 36, 0.2); color: #fbbf24;">
    <span style="font-size: 1.5rem; margin-right: 8px;">⚠️</span>
    <div>
        <strong>This customer is at MODERATE RISK of churn</strong><br/>
        <span style="font-size: 0.95rem;">Churn Probability: <strong>{probability * 100:.2f}%</strong></span>
    </div>
</div>"""
        else:
            risk_color = "#f87171"
            bar_gradient = "linear-gradient(90deg, #ef4444 0%, #b91c1c 100%)"
            alert_html = f"""<div class="alert-box alert-danger">
    <span style="font-size: 1.5rem; margin-right: 8px;">⚠️</span>
    <div>
        <strong>This customer is LIKELY TO CHURN</strong><br/>
        <span style="font-size: 0.95rem;">Churn Probability: <strong>{probability * 100:.2f}%</strong></span>
    </div>
</div>"""
            
        result_html = f"""<div class="result-card">
    <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">
        <div style="flex: 2; min-width: 250px;">
            <h3 style="margin-top: 0; color: #fff;">📊 Prediction Result</h3>
            {alert_html}
        </div>
        <div style="flex: 3; min-width: 300px;">
            <h3 style="margin-top: 0; color: #fff;">📈 Risk Score Meter</h3>
            <div style="margin-bottom: 8px; font-size: 1.1rem; color: #d1d5db;">
                Churn Risk Level: <strong style="color: {risk_color};">{probability * 100:.2f}%</strong>
            </div>
            <div class="risk-bar-bg">
                <div class="risk-bar-fill" style="width: {probability * 100}%; background: {bar_gradient};"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #9ca3af; margin-top: 5px;">
                <span>Low Risk (0%)</span>
                <span>Medium Risk (50%)</span>
                <span>High Risk (100%)</span>
            </div>
        </div>
    </div>
</div>"""
        st.markdown(result_html, unsafe_allow_html=True)

# Tabs section for plots and analytics
st.markdown("### 🔍 Model Insights & Analysis")
tab1, tab2, tab3 = st.tabs(["Model Comparison", "SHAP Explanation", "EDA Plots"])

with tab1:
    st.markdown("#### Classifier Performance Comparison")
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        if os.path.exists("plots/model_comparison.png"):
            st.image("plots/model_comparison.png", caption="Model Performance Metrics Comparison", use_container_width=True)
        else:
            st.info("Comparison plot not found. Run main.py to generate it.")
            
    with col_plot2:
        if os.path.exists("plots/roc_curves.png"):
            st.image("plots/roc_curves.png", caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
        else:
            st.info("ROC curve plot not found. Run main.py to generate it.")

with tab2:
    st.markdown("#### Feature Explanations using SHAP (Shapley Additive exPlanations)")
    if os.path.exists("plots/shap_summary.png"):
        st.image("plots/shap_summary.png", caption="SHAP Summary Plot (XGBoost)", use_container_width=True)
        st.markdown("""
            **Interpretation of SHAP Summary Plot:**
            *   **Feature Importance:** Features are ranked by default from top to bottom based on their predictive impact.
            *   **Feature Values:** Red indicates high value for a feature, and blue indicates low value.
            *   **Impact on Churn Risk:** Points plotted to the right of the zero line represent positive contributions to churn probability (higher risk), while points to the left represent negative contributions (lower risk/retention).
            *   *For example:* High tenure (red) shifts the points to the left, meaning longer-term customers are much less likely to churn. High monthly charges (red) shift points to the right, signifying higher risk.
        """)
    else:
        st.info("SHAP summary plot not found. Run main.py to generate it.")

with tab3:
    st.markdown("#### Dataset Exploratory Data Analysis Plots")
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        if os.path.exists("plots/churn_distribution.png"):
            st.image("plots/churn_distribution.png", caption="Class Balance: Churned vs Retained", use_container_width=True)
        else:
            st.info("Plot not found. Run main.py.")
            
        if os.path.exists("plots/tenure_by_churn.png"):
            st.image("plots/tenure_by_churn.png", caption="Customer Tenure Distribution vs Churn Status", use_container_width=True)
        else:
            st.info("Plot not found. Run main.py.")
            
    with col_eda2:
        if os.path.exists("plots/monthly_charges_by_churn.png"):
            st.image("plots/monthly_charges_by_churn.png", caption="Density of Monthly Charges by Churn Status", use_container_width=True)
        else:
            st.info("Plot not found. Run main.py.")
            
        if os.path.exists("plots/correlation_heatmap.png"):
            st.image("plots/correlation_heatmap.png", caption="Correlation Heatmap (Numerical Features)", use_container_width=True)
        else:
            st.info("Plot not found. Run main.py.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #71717a; font-size: 0.9rem;'>"
    "Dataset: Kaggle Telco Churn"
    "</div>", 
    unsafe_allow_html=True
)
