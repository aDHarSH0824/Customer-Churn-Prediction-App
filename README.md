# Customer Churn Prediction using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-darkgreen.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict whether a telecommunications customer will churn (leave = 1) or stay (0) using supervised machine learning algorithms, coupled with SHAP explainability and an interactive Streamlit web dashboard.

## 🚀 Live Demo
Run the interactive dashboard locally:
```bash
streamlit run app.py
```

## 🛠️ Tech Stack
*   **Programming Language:** Python
*   **Data Processing:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Model Explainability:** SHAP (Shapley Additive exPlanations)
*   **Dashboard Web App:** Streamlit
*   **Visualizations:** Matplotlib, Seaborn
*   **Serialization:** Joblib

## 📁 Project Structure
```text
churn_prediction/
├── data_loader.py        # load and clean dataset
├── eda.py                # EDA plots saved as PNG
├── preprocess.py         # feature engineering + train-test split
├── train.py              # train 3 models + save as .pkl files
├── evaluate.py           # metrics, confusion matrix, ROC curve
├── explain.py            # SHAP explainability
├── main.py               # runs the full ML pipeline
├── app.py                # Streamlit web app
├── requirements.txt      # all libraries
├── README.md             # GitHub description
├── models/               # saved .pkl model files (auto-created)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── plots/                # all PNG plots (auto-created)
    ├── churn_distribution.png
    ├── monthly_charges_by_churn.png
    ├── tenure_by_churn.png
    ├── correlation_heatmap.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_xgboost.png
    ├── model_comparison.png
    ├── roc_curves.png
    └── shap_summary.png
```

## ⚙️ How to Run

### Step 1: Install Dependencies
Create a virtual environment and install all packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run ML Pipeline
Train the classifiers, compute metrics, and export artifacts:
```bash
python main.py
```
*Note: This will download the dataset, perform EDA, scale features, train all 3 models, and produce all evaluation plots in `plots/` and weights in `models/`.*

### Step 3: Launch Web App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 🧠 Methodology
1.  **EDA (Exploratory Data Analysis):** Visualizing feature distributions, tenure durations, and correlations using Seaborn and Matplotlib.
2.  **Preprocessing & Encoding:** Handling missing charges, mapping binary features with `LabelEncoder`, one-hot encoding multi-class columns with `pd.get_dummies()`, and normalizing numerical features with `StandardScaler`.
3.  **Model Training:** Fitting Logistic Regression, Random Forest, and XGBoost Classifiers.
4.  **Evaluation:** Comparing F1, Accuracy, and ROC-AUC scores, and checking confusion matrices and ROC curves.
5.  **Explainability:** Computing SHAP tree explanations on the XGBoost model to understand individual and global feature impacts.
6.  **Deployment:** Packaging the pipeline outputs into a modern Streamlit application.

---

## 📊 Model Evaluation Results

After running the pipeline, the comparative scores are as follows:

| Model | Accuracy | F1 Score (Weighted) | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | `[accuracy_val]` | `[f1_val]` | `[auc_val]` |
| **Random Forest** | `[accuracy_val]` | `[f1_val]` | `[auc_val]` |
| **XGBoost Classifier** | `[accuracy_val]` | `[f1_val]` | `[auc_val]` |

*(Actual metrics will be printed upon running `python main.py`)*

---

## 📈 Key Findings from SHAP Explainability
The top 3 factors driving customer churn based on SHAP calculations are:
1.  **Contract Type (Month-to-month):** Month-to-month contracts have the highest positive impact on churn risk.
2.  **Tenure:** Customers with short tenure are significantly more likely to churn, while high tenure is a strong retention indicator.
3.  **Internet Service (Fiber Optic):** Subscribing to Fiber Optic internet is associated with higher churn rates, potentially due to pricing or service stability issues.

*(Run `explain.py` or check `plots/shap_summary.png` to review the visual distribution)*

---

## 💻 App Features
The Streamlit application includes 5 comprehensive sections:
1.  **Sidebar Project Description:** Background info, dataset details, and baseline model specs.
2.  **Interactive Demographic Inputs:** Real-time collection of gender, senior status, partner presence, dependents, and tenure.
3.  **Service Configuration:** Dynamic input selection of phone and internet services, online security, backup, tech support, and streaming.
4.  **Billing & Contract sliders:** Inputs for contract type, paperless billing, payment method, monthly charges, and total charges.
5.  **Predictive Model Output:** A visual probability bar gauge, custom success/warning cards, and tabs showing the comparative model charts, SHAP summary explanations, and EDA plots.
