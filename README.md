# PayGuard - Fraud Detection System

Production-grade fraud detection on 590,540 real transactions.

## Results
- PR-AUC: 0.8401
- ROC-AUC: 0.9695
- Recall: 81.3% (catches 4 out of 5 fraudsters)
- Dataset: IEEE-CIS Fraud Detection (Kaggle)

## Tech Stack
Python, XGBoost, LightGBM, SHAP, Streamlit, Scikit-learn, Pandas

## Pipeline
- 10 notebooks covering EDA, feature engineering, modeling, evaluation
- Streamlit dashboard for live fraud scoring
- SHAP explainability
- Cost-optimized threshold selection
