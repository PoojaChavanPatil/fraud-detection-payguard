# ============================================================
# PAYGUARD — Streamlit App
# Run: streamlit run app/payguard_app.py
# ============================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="PayGuard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(ROOT, 'models/xgb_tuned.pkl'))
    explainer = joblib.load(os.path.join(ROOT, 'models/shap_explainer.pkl'))
    scaler = joblib.load(os.path.join(ROOT, 'models/scaler.pkl'))
    config = pd.read_csv(os.path.join(ROOT, 'data/processed/threshold_config.csv')).iloc[0]
    metrics = pd.read_csv(os.path.join(ROOT, 'data/processed/final_metrics.csv')).iloc[0]

    feature_names = model.get_booster().feature_names
    # A "typical average transaction" baseline, built from the scaler's stored
    # per-feature means (saved when the model was trained) — no raw dataset needed.
    means = pd.Series(scaler.mean_, index=scaler.feature_names_in_).reindex(feature_names)
    baseline_row = pd.DataFrame([means.values], columns=feature_names)

    return model, explainer, config, metrics, baseline_row


st.title("🛡️ PayGuard — Fraud Detection Dashboard")
st.caption("IEEE-CIS Dataset · XGBoost · Production Pipeline")
st.divider()

try:
    model, explainer, config, metrics, baseline_row = load_assets()
    thresh = float(config['optimal_threshold_f2'])
    feature_names = model.get_booster().feature_names
    loaded = True
except Exception as e:
    st.warning(f"Run notebooks 01–10 first to train the model. ({e})")
    loaded = False
    thresh = 0.5

tab1, tab2, tab3 = st.tabs(["📊 Performance", "🔍 Score Transaction", "📋 About"])

# ── TAB 1: PERFORMANCE ────────────────────────────────────
with tab1:
    if loaded:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
        c2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        c3.metric("Recall", f"{metrics['recall']:.4f}")
        c4.metric("F2 Score", f"{metrics['f2']:.4f}")
        st.divider()
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Results at Optimal Threshold")
            st.metric("Fraud Caught", f"{int(metrics['tp']):,}  ({metrics['recall']*100:.1f}%)")
            st.metric("Fraud Missed", f"{int(metrics['fn']):,}")
            st.metric("False Alarms", f"{int(metrics['fp']):,}")
            st.metric("Expected Cost", f"${metrics['cost']:,.0f}")
        with col_r:
            try:
                st.image(os.path.join(ROOT, 'reports/final_evaluation.png'))
            except Exception:
                st.info("Run notebook 10 to generate charts")
    else:
        st.info("Train the model to see results here.")

# ── TAB 2: SCORE A TRANSACTION ────────────────────────────
with tab2:
    st.header("Score a Transaction in Real-Time")
    st.caption(
        "Starts from a typical average transaction (all 258 model features, "
        "using the training set's average values), then applies your inputs below "
        "on top of it — so the model sees a realistic, in-distribution transaction "
        "rather than a mostly-empty one."
    )
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Amount ($)", value=150.0, min_value=0.01)
        hour = st.slider("Hour (0-23)", 0, 23, 14)
        is_mobile = st.checkbox("Mobile transaction")
        is_intl = st.checkbox("International")
    with c2:
        email = st.selectbox("Email Domain",
                              ["gmail.com", "hotmail.com", "yahoo.com",
                               "mail.com", "outlook.es", "outlook.com", "other"])
        card_type = st.selectbox("Card Type", ["credit", "debit"])
        day = st.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    st.subheader("Risk Flags")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Micro (<$10)", "🚨 YES" if amount < 10 else "✅ NO")
    fc2.metric("Peak Hour (7am)", "🚨 YES" if hour == 7 else "✅ NO")
    fc3.metric("High-Risk Email", "🚨 YES" if email in ['mail.com', 'outlook.es'] else "✅ NO")

    if st.button("🛡️ Score This Transaction", type="primary", use_container_width=True) and loaded:
        fv = baseline_row.copy()
        day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

        def s(col, val):
            if col in fv.columns:
                fv[col] = val

        s('TransactionAmt', amount)
        s('amt_log', np.log1p(amount))
        s('hour', hour)
        s('day_of_week', day_map[day])
        s('is_micro_txn', int(amount < 10))
        s('is_high_value', int(amount > 500))
        s('is_night', int(hour <= 6))
        s('is_peak_fraud_hour', int(hour == 7))
        s('is_mobile', int(is_mobile))
        s('is_international', int(is_intl))
        s('p_email_high_risk', int(email in ['mail.com', 'outlook.es', 'aim.com']))
        s('is_weekend', int(day_map[day] >= 5))

        fv = fv[feature_names]

        score = model.predict_proba(fv)[0, 1]
        is_fraud = score >= thresh

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Score", f"{score:.4f}  ({score*100:.1f}%)")
        r2.metric("Decision", "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE")
        r3.metric("Threshold", f"{thresh:.2f}")

        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh([0], [score], color='crimson' if is_fraud else 'steelblue', height=0.5)
        ax.barh([0], [1 - score], left=[score], color='#e8e8e8', height=0.5)
        ax.axvline(x=thresh, color='black', linestyle='--', linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Fraud Probability')
        plt.tight_layout()
        st.pyplot(fig)

        # ── SHAP explanation ──────────────────────────
        st.divider()
        st.subheader("🔍 Why did the model say this?")
        shap_values = explainer.shap_values(fv)
        row_shap = shap_values[0]

        expl_df = pd.DataFrame({
            'feature': feature_names,
            'value': fv.iloc[0].values,
            'shap_value': row_shap
        })
        expl_df['abs_shap'] = expl_df['shap_value'].abs()
        top_features = expl_df.sort_values('abs_shap', ascending=False).head(10)
        top_features = top_features.sort_values('shap_value')

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors = ['crimson' if v > 0 else 'steelblue' for v in top_features['shap_value']]
        ax2.barh(top_features['feature'], top_features['shap_value'], color=colors)
        ax2.axvline(x=0, color='black', linewidth=0.8)
        ax2.set_xlabel('Contribution to fraud score (SHAP value)')
        ax2.set_title('Top 10 features driving this decision')
        plt.tight_layout()
        st.pyplot(fig2)

        st.caption(
            "🔴 Red bars push the score toward FRAUD · 🔵 Blue bars push toward LEGITIMATE. "
            "Some of these are engineered features from the anonymized IEEE-CIS dataset "
            "(V-columns, C-columns) that the model found most predictive during training — "
            "they stay fixed at their dataset-average value here since this demo doesn't "
            "have access to a live transaction's raw anonymized fields."
        )

# ── TAB 3: ABOUT ──────────────────────────────────────────
with tab3:
    st.markdown("""
    ## About PayGuard
    **Dataset**: IEEE-CIS Fraud Detection — 590,540 transactions, 3.5% fraud rate

    | Notebook | Task |
    |----------|------|
    | 01 | Data loading & merge |
    | 02 | Deep EDA |
    | 03 | Missing data strategy |
    | 04 | Feature engineering (39 features) |
    | 05 | Feature selection |
    | 06 | Imbalance strategy |
    | 07 | Baseline models (LR, NB) |
    | 08 | XGBoost + LightGBM + Optuna |
    | 09 | Threshold optimization |
    | 10 | Final evaluation + SHAP |

    **How the live scoring tab works**
    The model was trained on 258 features. This demo lets you control 12
    business-interpretable ones (amount, hour, mobile, email domain, etc.).
    The rest are held at their training-set average, so the model sees a
    realistic transaction shape rather than mostly-empty input. SHAP shows
    exactly which of all 258 features — including ones you don't directly
    control — drove each decision.

    **Key Findings**
    - mail.com email → 18.96% fraud rate (5× average)
    - Micro transactions <$10 → 7.77% fraud (card testing)
    - Mobile → 10.17% fraud vs 6.52% desktop
    - V45 strongest single feature (separation=1.54)
    - 7am peak fraud hour (10.61%)
    """)
