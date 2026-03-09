# ============================================================
# PAYGUARD — NOTEBOOK 05: Feature Selection
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\05_feature_selection.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)

print("=" * 55)
print("   NOTEBOOK 05 — Feature Selection")
print("=" * 55)

# ── LOAD WITH RAM FIX ──────────────────────────────────────
print("\n[1/5] Loading data (RAM optimized)...")
df = pd.read_csv('data/processed/train_engineered.csv', low_memory=False)

# Downcast floats to save memory
for col in df.select_dtypes(include='float64').columns:
    df[col] = df[col].astype('float32')
for col in df.select_dtypes(include='int64').columns:
    df[col] = df[col].astype('int32')

print(f"  ✓ Shape : {df.shape[0]:,} x {df.shape[1]}")
print(f"  ✓ Memory: {df.memory_usage().sum()/1024**2:.0f} MB")

# ── PREPARE FEATURES ───────────────────────────────────────
print("\n[2/5] Preparing feature matrix...")
drop_cols = ['TransactionID','TransactionDT','isFraud',
             'P_emaildomain','R_emaildomain','DeviceInfo','DeviceType',
             'card1_card2','device_os','ProductCD',
             'card4','card6','M1','M2','M3','M4','M5','M6','M7','M8','M9']

feature_cols = [c for c in df.columns if c not in drop_cols]
df_model     = df[feature_cols + ['isFraud']].copy()

# Encode any remaining object columns
for col in df_model.select_dtypes(include='object').columns:
    if col != 'isFraud':
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str)).astype('int32')

df_model.fillna(0, inplace=True)
X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud'].astype(int)

print(f"  ✓ Feature matrix: {X.shape[0]:,} x {X.shape[1]}")

# ── REMOVE HIGH CORRELATION ────────────────────────────────
print("\n[3/5] Removing highly correlated features...")
print("  Computing correlation matrix (takes ~2 mins)...")

# Sample 50k rows for correlation to save memory
sample_idx  = np.random.choice(len(X), size=min(50000, len(X)), replace=False)
X_sample    = X.iloc[sample_idx]
corr_matrix = X_sample.corr().abs()
upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr   = [col for col in upper.columns if any(upper[col] > 0.95)]
X_filtered  = X.drop(columns=high_corr)

print(f"  ✓ Removed {len(high_corr)} highly correlated columns")
print(f"  ✓ Features remaining: {X_filtered.shape[1]}")

# Free memory
del corr_matrix, upper, X_sample
import gc; gc.collect()

# ── RANDOM FOREST IMPORTANCE ───────────────────────────────
print("\n[4/5] Random Forest feature importance...")
print("  Training on 100k sample (takes 3-5 mins)...")

# Use 100k sample to fit RF — enough for importance ranking
sample_idx2  = np.random.choice(len(X_filtered), size=min(100000, len(X_filtered)), replace=False)
X_rf         = X_filtered.iloc[sample_idx2]
y_rf         = y.iloc[sample_idx2]

X_tr, X_val, y_tr, y_val = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)

rf = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_tr, y_tr)

importance_df = pd.DataFrame({
    'feature':    X_filtered.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 20 features:")
print(importance_df.head(20).to_string(index=False))

# Plot
plt.figure(figsize=(12, 8))
top30  = importance_df.head(30)
colors = ['crimson' if i < 10 else 'steelblue' for i in range(len(top30))]
plt.barh(top30['feature'][::-1], top30['importance'][::-1], color=colors[::-1])
plt.title('Top 30 Features — Random Forest Importance', fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=150)
plt.close()
print("  ✓ reports/feature_importance.png saved")

# ── SELECT & SAVE ──────────────────────────────────────────
print("\n[5/5] Selecting features and saving...")

threshold = importance_df['importance'].quantile(0.25)
selected  = importance_df[importance_df['importance'] > threshold]['feature'].tolist()

must_keep = [
    'amt_zscore','amt_log','is_micro_txn','is_peak_fraud_hour',
    'p_email_fraud_rate','r_email_fraud_rate','card4_fraud_rate',
    'card6_fraud_rate','is_mobile','has_device','device_os_fraud_rate',
    'card_multi_email','is_international','hour','card1_txn_count'
]
for f in must_keep:
    if f in X_filtered.columns and f not in selected:
        selected.append(f)

pd.Series(selected).to_csv('data/processed/selected_features.csv', index=False, header=False)

# Save final dataset — use only selected features
final_cols = selected + ['isFraud']
df_final   = df_model[[c for c in final_cols if c in df_model.columns]].copy()
df_final.to_csv('data/processed/train_final.csv', index=False)

print(f"\n{'='*55}")
print(f"   NOTEBOOK 05 COMPLETE")
print(f"{'='*55}")
print(f"  Features before : {X_filtered.shape[1]}")
print(f"  Features after  : {len(selected)}")
print(f"  Final shape     : {df_final.shape[0]:,} x {df_final.shape[1]}")
print(f"  Fraud rate      : {df_final['isFraud'].mean()*100:.3f}%")
print(f"  Saved           : data/processed/train_final.csv ✓")
print(f"{'='*55}")
