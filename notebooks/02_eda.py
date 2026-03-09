# ============================================================
# PAYGUARD — NOTEBOOK 02: Deep EDA
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\02_eda.py
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)

print("=" * 55)
print("   NOTEBOOK 02 — Deep EDA")
print("=" * 55)

# ── LOAD ───────────────────────────────────────────────────
print("\n[1/6] Loading merged data...")
df = pd.read_csv('data/processed/train_merged.csv', low_memory=False)
print(f"  ✓ {df.shape[0]:,} rows x {df.shape[1]} cols | Fraud: {df['isFraud'].mean()*100:.3f}%")

# ── EMAIL DOMAINS ──────────────────────────────────────────
print("\n[2/6] Email domain analysis...")
email_stats = df.groupby('P_emaildomain').agg(
    total=('isFraud','count'),
    fraud=('isFraud','sum')
).query('total >= 100')
email_stats['fraud_rate'] = email_stats['fraud'] / email_stats['total'] * 100
email_stats = email_stats.sort_values('fraud_rate', ascending=False)

print("  Top 10 riskiest email domains:")
print(email_stats.head(10)[['total','fraud','fraud_rate']].round(2).to_string())

top10 = email_stats.head(10)
plt.figure(figsize=(12, 5))
colors = ['crimson' if r > 10 else 'orange' if r > 5 else 'steelblue'
          for r in top10['fraud_rate']]
plt.bar(top10.index, top10['fraud_rate'], color=colors)
plt.axhline(y=df['isFraud'].mean()*100, color='black', linestyle='--', label='Average')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Fraud Rate (%)')
plt.title('Fraud Rate by Email Domain (min 100 transactions)', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('reports/fraud_by_email.png', dpi=150)
plt.close()
print("  ✓ reports/fraud_by_email.png")

# ── DAY OF WEEK ────────────────────────────────────────────
print("\n[3/6] Day of week analysis...")
df['hour']       = (df['TransactionDT'] // 3600) % 24
df['day_of_week']= (df['TransactionDT'] // (3600*24)) % 7
day_names        = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day_name']   = df['day_of_week'].map(day_names)

day_fraud = df.groupby('day_name')['isFraud'].mean() * 100
day_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
day_fraud = day_fraud.reindex(day_order)
print("  Fraud rate by day:")
for d, r in day_fraud.items():
    print(f"    {d}: {r:.2f}%")

plt.figure(figsize=(9, 4))
plt.bar(day_fraud.index, day_fraud.values, color='steelblue')
plt.axhline(y=df['isFraud'].mean()*100, color='red', linestyle='--', label='Average')
plt.ylabel('Fraud Rate (%)')
plt.title('Fraud Rate by Day of Week', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('reports/fraud_by_day.png', dpi=150)
plt.close()
print("  ✓ reports/fraud_by_day.png")

# ── V-FEATURES ─────────────────────────────────────────────
print("\n[4/6] V-feature separation analysis (takes ~2 mins)...")
v_cols = [c for c in df.columns if c.startswith('V')]
fraud_df = df[df['isFraud']==1]
legit_df = df[df['isFraud']==0]

sep_scores = {}
for col in v_cols:
    f_mean = fraud_df[col].mean()
    l_mean = legit_df[col].mean()
    std    = df[col].std()
    if std > 0 and not np.isnan(f_mean) and not np.isnan(l_mean):
        sep_scores[col] = abs(f_mean - l_mean) / std

sep_series = pd.Series(sep_scores).sort_values(ascending=False)
print("  Top 15 V-features by separation score:")
print(sep_series.head(15).round(4).to_string())

# ── NON-V CORRELATIONS ─────────────────────────────────────
print("\n[5/6] Non-V feature correlations...")
non_v_cols = [c for c in df.select_dtypes(include=[np.number]).columns
              if not c.startswith('V') and c != 'isFraud']
corr = df[non_v_cols + ['isFraud']].corr()['isFraud'].drop('isFraud')
corr_abs = corr.abs().sort_values(ascending=False)
print("  Top 10 correlated non-V features:")
print(corr_abs.head(10).round(4).to_string())

plt.figure(figsize=(10, 5))
top15c = corr_abs.head(15)
colors = ['crimson' if corr[i] > 0 else 'steelblue' for i in top15c.index]
plt.bar(top15c.index, top15c.values, color=colors)
plt.xticks(rotation=30, ha='right')
plt.ylabel('|Correlation with isFraud|')
plt.title('Top 15 Non-V Feature Correlations\n(Red=Positive, Blue=Negative)', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/feature_correlation.png', dpi=150)
plt.close()
print("  ✓ reports/feature_correlation.png")

# ── AMOUNT BUCKETS ─────────────────────────────────────────
print("\n[6/6] Amount bucket analysis...")
bins   = [0, 10, 25, 50, 100, 200, 500, 1000, 99999]
labels = ['<10','10-25','25-50','50-100','100-200','200-500','500-1k','>1k']
df['amt_bucket'] = pd.cut(df['TransactionAmt'], bins=bins, labels=labels)
bucket_fraud = df.groupby('amt_bucket', observed=True)['isFraud'].mean() * 100
print("  Fraud rate by amount bucket:")
for b, r in bucket_fraud.items():
    flag = ' ← CARD TESTING' if b == '<10' else ''
    print(f"    {str(b):10s}: {r:.2f}%{flag}")

# ── SAVE EDA FEATURES ──────────────────────────────────────
top_v = sep_series.head(50).index.tolist()
pd.Series(top_v).to_csv('data/processed/top_v_features.csv', index=False, header=False)
df[['TransactionID','hour','day_of_week','day_name','amt_bucket']].to_csv(
    'data/processed/eda_features.csv', index=False)

print(f"\n{'='*55}")
print(f"   NOTEBOOK 02 COMPLETE")
print(f"{'='*55}")
print(f"  mail.com fraud rate : {email_stats.loc['mail.com','fraud_rate']:.2f}% (if present)")
print(f"  Top V-feature       : {sep_series.index[0]} (score={sep_series.iloc[0]:.4f})")
print(f"  Top non-V feature   : {corr_abs.index[0]} (corr={corr_abs.iloc[0]:.4f})")
print(f"  Saved top_v_features.csv & eda_features.csv ✓")
print(f"{'='*55}")
