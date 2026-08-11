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
    config = pd.read_csv(os.path.join(ROOT, 'data/processed/threshold_config.csv')).iloc[0]
    metrics = pd.read_csv(os.path.join(ROOT, 'data/processed/final_metrics.csv')).iloc[0]
    demo_samples = pd.read_csv(os.path.join(ROOT, 'data/processed/demo_samples.csv'))
    return model, explainer, config, metrics, demo_samples


st.title("🛡️ PayGuard — Fraud Detection Dashboard")
st.caption("IEEE-CIS Dataset · XGBoost · Production Pipeline")
st.divider()

try:
    model, explainer, config, metrics, demo_samples = load_assets()
    thresh = float(config['optimal_threshold_f2'])
    feature_names = model.get_booster().feature_names
    loaded = True
except Exception as e:
    st.warning(f"Run notebooks 01–11 first to train the model and export demo samples. ({e})")
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
    st.header("Score a Real Transaction")
    st.caption(
        "This starts from a REAL held-out transaction (all 258 model features), "
        "so the fraud signal is realistic — not just the handful of fields you can edit below."
    )

    if loaded:
        if "demo_row_idx" not in st.session_state:
            st.session_state.demo_row_idx = int(np.random.randint(0, len(demo_samples)))

        if st.button("🔀 Load a different random transaction"):
            st.session_state.demo_row_idx = int(np.random.randint(0, len(demo_samples)))

        base_row = demo_samples.iloc[[st.session_state.demo_row_idx]].copy()
        true_label = int(base_row['true_label'].iloc[0]) if 'true_label' in base_row.columns else None
        base_row = base_row[feature_names]  # drop true_label, enforce model column order

        if true_label is not None:
            st.caption(f"Base transaction #{st.session_state.demo_row_idx} — "
                       f"true label in dataset: {'🚨 FRAUD' if true_label == 1 else '✅ LEGITIMATE'} "
                       f"(hidden from the model — shown here for your reference only)")

        c1, c2 = st.columns(2)
        with c1:
            default_amt = float(base_row['TransactionAmt'].iloc[0]) if 'TransactionAmt' in base_row.columns else 150.0
            default_hour = int(base_row['hour'].iloc[0]) if 'hour' in base_row.columns else 14
            amount = st.number_input("Amount ($)", value=round(max(default_amt, 0.01), 2), min_value=0.01)
            hour = st.slider("Hour (0-23)", 0, 23, min(max(default_hour, 0), 23))
            is_mobile = st.checkbox("Mobile transaction", value=bool(base_row.get('is_mobile', pd.Series([0])).iloc[0]))
            is_intl = st.checkbox("International", value=bool(base_row.get('is_international', pd.Series([0])).iloc[0]))
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

        if st.button("🛡️ Score This Transaction", type="primary", use_container_width=True):
            fv = base_row.copy()
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
                "This is the same SHAP explainability used in the notebook pipeline (10_evaluation.py), "
                "applied live to this transaction."
            )
    else:
        st.info("Train the model and export demo samples to enable scoring.")

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
    | 11 | Export demo samples for live scoring |

    **How the live scoring tab works**
    Each "transaction" you score starts from a real, held-out row from the test set —
    all 258 features the model was trained on, not just the ones shown in the UI.
    You can adjust amount, hour, mobile/international flags, and email domain on top of
    that real row, then the model scores it and SHAP explains exactly which features
    pushed the score up or down.

    **Key Findings**
    - mail.com email → 18.96% fraud rate (5× average)
    - Micro transactions <$10 → 7.77% fraud (card testing)
    - Mobile → 10.17% fraud vs 6.52% desktop
    - V45 strongest single feature (separation=1.54)
    - 7am peak fraud hour (10.61%)
    """)
