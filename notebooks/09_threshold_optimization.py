# ============================================================
# PAYGUARD — NOTEBOOK 09: Threshold Optimization
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\09_threshold_optimization.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, fbeta_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)

print("=" * 55)
print("   NOTEBOOK 09 — Threshold Optimization")
print("=" * 55)

print("\n[1/3] Loading model and data...")
chunks = []
for chunk in pd.read_csv('data/processed/train_final.csv',
                          chunksize=50000, low_memory=False):
    for col in chunk.select_dtypes('float64').columns:
        chunk[col] = chunk[col].astype('float32')
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

X = df.drop('isFraud', axis=1).fillna(0)
y = df['isFraud'].astype(int)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = joblib.load('models/xgb_tuned.pkl')
proba = model.predict_proba(X_test)[:,1]
print(f"  ✓ {len(proba):,} test predictions ready")

FN_COST = 150
FP_COST = 5

print("\n[2/3] Scanning thresholds...")
thresholds = np.arange(0.01, 0.99, 0.01)
f2_scores, costs, precisions, recalls = [], [], [], []

for t in thresholds:
    preds = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f2   = fbeta_score(y_test, preds, beta=2, zero_division=0)
    cost = fn * FN_COST + fp * FP_COST
    f2_scores.append(f2); costs.append(cost)
    precisions.append(prec); recalls.append(rec)

best_f2_t   = thresholds[np.argmax(f2_scores)]
best_cost_t = thresholds[np.argmin(costs)]
idx50       = np.argmin(np.abs(thresholds - 0.50))

print(f"  Default (0.50)     → F2: {f2_scores[idx50]:.4f}  Cost: ${costs[idx50]:,.0f}")
print(f"  Best F2 ({best_f2_t:.2f})   → F2: {max(f2_scores):.4f}  Recall: {recalls[np.argmax(f2_scores)]:.4f}")
print(f"  Min Cost ({best_cost_t:.2f}) → Cost: ${min(costs):,.0f}  Saving: ${costs[idx50]-min(costs):,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
axes[0].plot(thresholds, recalls,    'r-', label='Recall',    linewidth=2)
axes[0].plot(thresholds, f2_scores,  'g-', label='F2 Score',  linewidth=2.5)
axes[0].axvline(x=best_f2_t, color='green', linestyle='--', label=f'Best F2={best_f2_t:.2f}')
axes[0].axvline(x=0.5,       color='gray',  linestyle=':',  label='Default=0.50')
axes[0].set_xlabel('Threshold'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title('Metrics vs Threshold', fontweight='bold')

axes[1].plot(thresholds, costs, 'crimson', linewidth=2.5)
axes[1].axvline(x=best_cost_t, color='green', linestyle='--', label=f'Min cost={best_cost_t:.2f}')
axes[1].axvline(x=0.5,         color='gray',  linestyle=':',  label='Default=0.50')
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Cost ($)')
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_title('Business Cost vs Threshold', fontweight='bold')

plt.suptitle('Threshold Optimization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/threshold_optimization.png', dpi=150)
plt.close()

print("\n[3/3] Saving config...")
pd.DataFrame([{
    'optimal_threshold_f2':   best_f2_t,
    'optimal_threshold_cost': best_cost_t,
    'fn_cost': FN_COST, 'fp_cost': FP_COST,
    'best_f2_score':   max(f2_scores),
    'best_recall':     recalls[np.argmax(f2_scores)],
    'best_precision':  precisions[np.argmax(f2_scores)],
    'cost_saving':     costs[idx50] - min(costs)
}]).to_csv('data/processed/threshold_config.csv', index=False)

print(f"\n{'='*55}")
print(f"   NOTEBOOK 09 COMPLETE")
print(f"{'='*55}")
print(f"  Optimal F2 threshold  : {best_f2_t:.2f}")
print(f"  Optimal cost threshold: {best_cost_t:.2f}")
print(f"  Cost saving vs default: ${costs[idx50]-min(costs):,.0f}")
print(f"  Saved: data/processed/threshold_config.csv ✓")
print(f"{'='*55}")
