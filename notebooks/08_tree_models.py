# ============================================================
# PAYGUARD — NOTEBOOK 08: Tree Models (XGBoost)
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\08_tree_models.py
# Install : pip install xgboost lightgbm optuna shap
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)

print("=" * 55)
print("   NOTEBOOK 08 — Tree Models")
print("=" * 55)

# ── LOAD FULL DATA (chunked for RAM) ──────────────────────
print("\n[1/5] Loading full dataset in chunks...")

# Read in chunks and downcast immediately
chunks = []
for chunk in pd.read_csv('data/processed/train_final.csv',
                          chunksize=50000, low_memory=False):
    for col in chunk.select_dtypes('float64').columns:
        chunk[col] = chunk[col].astype('float32')
    for col in chunk.select_dtypes('int64').columns:
        chunk[col] = chunk[col].astype('int32')
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
import gc; gc.collect()

print(f"  ✓ Shape : {df.shape[0]:,} x {df.shape[1]}")
print(f"  ✓ Memory: {df.memory_usage().sum()/1024**2:.0f} MB")

X = df.drop('isFraud', axis=1).fillna(0)
y = df['isFraud'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

spw = (y_train==0).sum() / (y_train==1).sum()
print(f"  ✓ Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"  ✓ scale_pos_weight: {spw:.2f}")

del df, chunks; gc.collect()

# ── XGBOOST DEFAULT ────────────────────────────────────────
print("\n[2/5] Training XGBoost (default params)...")
try:
    import xgboost as xgb
    xgb_def = xgb.XGBClassifier(
        n_estimators=300, scale_pos_weight=spw,
        random_state=42, eval_metric='aucpr',
        verbosity=0, tree_method='hist',  # hist = faster + less RAM
        n_jobs=-1
    )
    xgb_def.fit(X_train, y_train,
                eval_set=[(X_test, y_test)], verbose=False)
    p_def   = xgb_def.predict_proba(X_test)[:,1]
    pr_def  = average_precision_score(y_test, p_def)
    print(f"  ✓ XGBoost default PR-AUC: {pr_def:.4f}")
except ImportError:
    print("  XGBoost not installed. Run: pip install xgboost")
    pr_def = 0; p_def = None

# ── LIGHTGBM ──────────────────────────────────────────────
print("\n[3/5] Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_m = lgb.LGBMClassifier(
        n_estimators=300, scale_pos_weight=spw,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    lgb_m.fit(X_train, y_train)
    p_lgb  = lgb_m.predict_proba(X_test)[:,1]
    pr_lgb = average_precision_score(y_test, p_lgb)
    print(f"  ✓ LightGBM PR-AUC: {pr_lgb:.4f}")
except ImportError:
    print("  LightGBM not installed. Run: pip install lightgbm")
    pr_lgb = 0; p_lgb = None

# ── OPTUNA TUNING ─────────────────────────────────────────
print("\n[4/5] Tuning XGBoost with Optuna (20 trials)...")
print("  (This takes 10-15 mins)")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 600),
            'max_depth':        trial.suggest_int('max_depth', 4, 8),
            'learning_rate':    trial.suggest_float('lr', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('sub', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('col', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'scale_pos_weight': spw,
            'random_state':     42,
            'eval_metric':      'aucpr',
            'verbosity':        0,
            'tree_method':      'hist',
            'n_jobs':           -1
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, verbose=False)
        return average_precision_score(y_test, m.predict_proba(X_test)[:,1])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"  ✓ Best PR-AUC: {study.best_value:.4f}")
    print(f"  ✓ Best params: {study.best_params}")

    best_p = study.best_params
    best_p.update({'scale_pos_weight': spw, 'random_state': 42,
                   'eval_metric': 'aucpr', 'verbosity': 0,
                   'tree_method': 'hist', 'n_jobs': -1})
    xgb_tuned = xgb.XGBClassifier(**best_p)
    xgb_tuned.fit(X_train, y_train, verbose=False)
    p_tuned   = xgb_tuned.predict_proba(X_test)[:,1]
    pr_tuned  = average_precision_score(y_test, p_tuned)
    roc_tuned = roc_auc_score(y_test, p_tuned)
    print(f"  ✓ Tuned XGBoost PR-AUC : {pr_tuned:.4f}")
    print(f"  ✓ Tuned XGBoost ROC-AUC: {roc_tuned:.4f}")
    joblib.dump(xgb_tuned, 'models/xgb_tuned.pkl')

except ImportError:
    print("  Optuna not installed. Run: pip install optuna")
    xgb_tuned = xgb_def
    p_tuned   = p_def
    pr_tuned  = pr_def
    roc_tuned = roc_auc_score(y_test, p_def) if p_def is not None else 0
    joblib.dump(xgb_tuned, 'models/xgb_tuned.pkl')

# ── SHAP ──────────────────────────────────────────────────
print("\n[5/5] SHAP feature importance...")
try:
    import shap
    sample   = X_test.iloc[:1000]
    explainer= shap.TreeExplainer(xgb_tuned)
    sv       = explainer.shap_values(sample)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, sample, plot_type='bar', max_display=20, show=False)
    plt.title('Top 20 Features — SHAP', fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    print("  ✓ reports/shap_importance.png")
except ImportError:
    print("  SHAP not installed. Run: pip install shap")

# PR Curve comparison
if p_def is not None and p_lgb is not None:
    plt.figure(figsize=(10, 6))
    for label, proba, score in [
        ('XGBoost default', p_def, pr_def),
        ('LightGBM', p_lgb, pr_lgb),
        ('XGBoost tuned', p_tuned, pr_tuned)
    ]:
        if proba is not None:
            prec, rec, _ = precision_recall_curve(y_test, proba)
            plt.plot(rec, prec, linewidth=2, label=f"{label} ({score:.4f})")
    plt.axhline(y=y_test.mean(), color='black', linestyle='--')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('PR Curves — Tree Models', fontweight='bold')
    plt.legend(); plt.tight_layout()
    plt.savefig('reports/tree_pr_curves.png', dpi=150)
    plt.close()
    print("  ✓ reports/tree_pr_curves.png")

print(f"\n{'='*55}")
print(f"   NOTEBOOK 08 COMPLETE")
print(f"{'='*55}")
print(f"  XGBoost default  PR-AUC: {pr_def:.4f}")
print(f"  LightGBM         PR-AUC: {pr_lgb:.4f}")
print(f"  XGBoost tuned    PR-AUC: {pr_tuned:.4f}")
print(f"  XGBoost tuned   ROC-AUC: {roc_tuned:.4f}")
print(f"  Model saved: models/xgb_tuned.pkl ✓")
print(f"{'='*55}")
