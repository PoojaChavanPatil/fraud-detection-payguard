# ============================================================
# PAYGUARD — NOTEBOOK 07: Baseline Models
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\07_baseline_models.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)

print("=" * 55)
print("   NOTEBOOK 07 — Baseline Models")
print("=" * 55)

# ── LOAD ───────────────────────────────────────────────────
print("\n[1/4] Loading data (80k rows)...")
df = pd.read_csv('data/processed/train_final.csv', nrows=80000)
df['isFraud'] = df['isFraud'].astype(int)
for col in df.select_dtypes('float64').columns:
    df[col] = df[col].astype('float32')

X = df.drop('isFraud', axis=1).fillna(0)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
Xtr    = scaler.fit_transform(X_train)
Xte    = scaler.transform(X_test)
print(f"  ✓ Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── TRAIN MODELS ──────────────────────────────────────────
print("\n[2/4] Training baseline models...")
results = {}

def evaluate(name, model, Xtr_, ytr_, Xte_, yte_):
    model.fit(Xtr_, ytr_)
    p       = model.predict_proba(Xte_)[:,1]
    pr_auc  = average_precision_score(yte_, p)
    roc_auc = roc_auc_score(yte_, p)
    results[name] = {'pr_auc': pr_auc, 'roc_auc': roc_auc, 'proba': p, 'model': model}
    print(f"  {name:35s} PR-AUC: {pr_auc:.4f}  ROC-AUC: {roc_auc:.4f}")

evaluate("Naive Bayes",             GaussianNB(), Xtr, y_train, Xte, y_test)
evaluate("LR L2 (balanced)",
    LogisticRegression(class_weight='balanced', max_iter=500, random_state=42),
    Xtr, y_train, Xte, y_test)
evaluate("LR L1 (balanced)",
    LogisticRegression(penalty='l1', solver='liblinear',
                       class_weight='balanced', max_iter=500, random_state=42),
    Xtr, y_train, Xte, y_test)
evaluate("LR L2 C=10 (balanced)",
    LogisticRegression(C=10, class_weight='balanced', max_iter=500, random_state=42),
    Xtr, y_train, Xte, y_test)

# ── PR CURVES ─────────────────────────────────────────────
print("\n[3/4] Plotting PR curves...")
plt.figure(figsize=(10, 6))
colors = ['gray','steelblue','darkorange','crimson']
for (name, res), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, res['proba'])
    plt.plot(rec, prec, color=color, linewidth=2,
             label=f"{name} ({res['pr_auc']:.4f})")
plt.axhline(y=y_test.mean(), color='black', linestyle='--',
            label=f"Random ({y_test.mean():.4f})")
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('PR Curves — Baseline Models', fontweight='bold')
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig('reports/baseline_pr_curves.png', dpi=150)
plt.close()
print("  ✓ reports/baseline_pr_curves.png")

# ── SAVE ──────────────────────────────────────────────────
print("\n[4/4] Saving best model...")
best_name  = max(results, key=lambda k: results[k]['pr_auc'])
best_model = results[best_name]['model']
joblib.dump(best_model, 'models/baseline_lr.pkl')

print(f"\n{'='*55}")
print(f"   NOTEBOOK 07 COMPLETE")
print(f"{'='*55}")
for n, r in results.items():
    print(f"  {n:35s} PR-AUC: {r['pr_auc']:.4f}")
print(f"\n  Best     : {best_name}")
print(f"  NOTE: XGBoost in NB08 will reach 0.80+ PR-AUC")
print(f"{'='*55}")
