# ============================================================
# PAYGUARD — NOTEBOOK 10: Final Evaluation
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\10_evaluation.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix, fbeta_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)

print("=" * 55)
print("   NOTEBOOK 10 — Final Evaluation")
print("=" * 55)

print("\n[1/5] Loading model, data, config...")
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

model   = joblib.load('models/xgb_tuned.pkl')
config  = pd.read_csv('data/processed/threshold_config.csv').iloc[0]
thresh  = float(config['optimal_threshold_f2'])
proba   = model.predict_proba(X_test)[:,1]
preds   = (proba >= thresh).astype(int)

print("\n[2/5] Computing metrics...")
pr_auc  = average_precision_score(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)
f2      = fbeta_score(y_test, preds, beta=2)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
fn_cost   = float(config['fn_cost'])
fp_cost   = float(config['fp_cost'])
total_cost= fn * fn_cost + fp * fp_cost

print(f"""
  ┌─────────────────────────────────────────┐
  │         PAYGUARD FINAL RESULTS          │
  ├─────────────────────────────────────────┤
  │  PR-AUC         : {pr_auc:.4f}                │
  │  ROC-AUC        : {roc_auc:.4f}                │
  │  F2 Score       : {f2:.4f}                │
  │  Precision      : {precision:.4f}                │
  │  Recall         : {recall:.4f}                │
  ├─────────────────────────────────────────┤
  │  Fraud caught   : {tp:,} TP              
  │  Fraud missed   : {fn:,} FN              
  │  False alarms   : {fp:,} FP              
  │  Correct legit  : {tn:,} TN              
  ├─────────────────────────────────────────┤
  │  Expected cost  : ${total_cost:,.0f}            
  └─────────────────────────────────────────┘
""")

print("\n[3/5] Confusion matrix plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = np.array([[tn, fp],[fn, tp]])
axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['Legit','Fraud'])
axes[0].set_yticklabels(['Legit','Fraud'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix', fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                     fontsize=13, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')

prec_c, rec_c, _ = precision_recall_curve(y_test, proba)
axes[1].plot(rec_c, prec_c, 'steelblue', linewidth=2.5, label=f'PR-AUC={pr_auc:.4f}')
axes[1].axhline(y=y_test.mean(), color='red', linestyle='--', label='Random')
axes[1].scatter([recall], [precision], color='red', s=100, zorder=5,
                label=f'Threshold={thresh:.2f}')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('PayGuard — Final Evaluation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/final_evaluation.png', dpi=150)
plt.close()
print("  ✓ reports/final_evaluation.png")

print("\n[4/5] Score distribution...")
plt.figure(figsize=(10, 5))
plt.hist(proba[y_test==0], bins=100, alpha=0.6, color='steelblue',
         label='Legitimate', density=True)
plt.hist(proba[y_test==1], bins=100, alpha=0.7, color='crimson',
         label='Fraud', density=True)
plt.axvline(x=thresh, color='black', linestyle='--', linewidth=2,
            label=f'Threshold={thresh:.2f}')
plt.xlabel('Fraud Score'); plt.ylabel('Density')
plt.title('Score Distribution', fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig('reports/score_distribution.png', dpi=150)
plt.close()
print("  ✓ reports/score_distribution.png")

print("\n[5/5] SHAP analysis...")
try:
    import shap
    explainer = joblib.load('models/shap_explainer.pkl')
    sv = explainer.shap_values(X_test.iloc[:500])
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test.iloc[:500], max_display=20, show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ reports/shap_beeswarm.png")
except Exception as e:
    print(f"  SHAP skipped: {e}")

pd.DataFrame([{
    'pr_auc': pr_auc, 'roc_auc': roc_auc, 'f2': f2,
    'precision': precision, 'recall': recall,
    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    'threshold': thresh, 'cost': total_cost
}]).to_csv('data/processed/final_metrics.csv', index=False)

print(f"\n{'='*55}")
print(f"   NOTEBOOK 10 COMPLETE")
print(f"{'='*55}")
print(f"  PR-AUC   : {pr_auc:.4f}")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"  Recall   : {recall:.4f}  ({tp:,} caught / {fn:,} missed)")
print(f"  Precision: {precision:.4f}  ({fp:,} false alarms)")
print(f"  Saved    : data/processed/final_metrics.csv ✓")
print(f"{'='*55}")
