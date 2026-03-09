# ============================================================
# PAYGUARD — NOTEBOOK 01: Data Loading & First Look
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\01_data_loading.py
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Always run from payguard root ──────────────────────────
ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports',        exist_ok=True)
os.makedirs('models',         exist_ok=True)

print("=" * 55)
print("   NOTEBOOK 01 — Data Loading")
print("=" * 55)

# ── LOAD RAW FILES ─────────────────────────────────────────
print("\n[1/5] Loading raw CSV files...")

# Load with low_memory=False to avoid dtype warnings
trans = pd.read_csv('data/raw/train_transaction.csv', low_memory=False)
ident = pd.read_csv('data/raw/train_identity.csv',    low_memory=False)

print(f"  Transactions : {trans.shape[0]:,} rows x {trans.shape[1]} cols")
print(f"  Identity     : {ident.shape[0]:,} rows x {ident.shape[1]} cols")

# ── MERGE ──────────────────────────────────────────────────
print("\n[2/5] Merging on TransactionID (left join)...")
df = trans.merge(ident, on='TransactionID', how='left')
print(f"  Merged shape : {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"  Identity match: {ident.shape[0]/trans.shape[0]*100:.1f}% of transactions")

# ── BASIC STATS ────────────────────────────────────────────
print("\n[3/5] Key statistics...")
fraud_count = df['isFraud'].sum()
fraud_rate  = df['isFraud'].mean() * 100
print(f"  Fraud count  : {fraud_count:,}")
print(f"  Legit count  : {(df['isFraud']==0).sum():,}")
print(f"  Fraud rate   : {fraud_rate:.3f}%")
print(f"  Avg fraud amt: ${df[df['isFraud']==1]['TransactionAmt'].mean():.2f}")
print(f"  Avg legit amt: ${df[df['isFraud']==0]['TransactionAmt'].mean():.2f}")

# Missing data summary
missing_pct = (df.isnull().sum() / len(df) * 100)
print(f"\n  Columns >90% missing : {(missing_pct > 90).sum()}")
print(f"  Columns >50% missing : {(missing_pct > 50).sum()}")
print(f"  Columns zero missing : {(missing_pct == 0).sum()}")

# ── CHARTS ─────────────────────────────────────────────────
print("\n[4/5] Saving charts...")

# Fraud by hour
df['hour'] = (df['TransactionDT'] // 3600) % 24
hour_fraud = df.groupby('hour')['isFraud'].mean() * 100
plt.figure(figsize=(12, 4))
bars = plt.bar(hour_fraud.index, hour_fraud.values,
               color=['crimson' if v > 5 else 'steelblue' for v in hour_fraud.values])
plt.axhline(y=fraud_rate, color='black', linestyle='--', label=f'Average ({fraud_rate:.2f}%)')
plt.xlabel('Hour of Day')
plt.ylabel('Fraud Rate (%)')
plt.title('Fraud Rate by Hour of Day', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('reports/fraud_by_hour.png', dpi=150)
plt.close()

# Amount distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
df[df['isFraud']==0]['TransactionAmt'].clip(0,500).hist(bins=50, color='steelblue', alpha=0.7)
plt.title('Legit Amount Distribution')
plt.xlabel('Amount ($)')
plt.subplot(1, 2, 2)
df[df['isFraud']==1]['TransactionAmt'].clip(0,500).hist(bins=50, color='crimson', alpha=0.7)
plt.title('Fraud Amount Distribution')
plt.xlabel('Amount ($)')
plt.tight_layout()
plt.savefig('reports/amount_distribution.png', dpi=150)
plt.close()

print("  ✓ reports/fraud_by_hour.png")
print("  ✓ reports/amount_distribution.png")

# ── SAVE ───────────────────────────────────────────────────
print("\n[5/5] Saving merged file...")
df.to_csv('data/processed/train_merged.csv', index=False)

print(f"\n{'='*55}")
print(f"   NOTEBOOK 01 COMPLETE")
print(f"{'='*55}")
print(f"  Merged shape : {df.shape[0]:,} x {df.shape[1]}")
print(f"  Fraud rate   : {fraud_rate:.3f}%")
print(f"  Saved        : data/processed/train_merged.csv ✓")
print(f"{'='*55}")
