# ============================================================
# PAYGUARD — NOTEBOOK 03: Missing Data Strategy
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\03_missing_data.py
# ============================================================

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)

print("=" * 55)
print("   NOTEBOOK 03 — Missing Data Strategy")
print("=" * 55)

# ── LOAD ───────────────────────────────────────────────────
print("\n[1/5] Loading merged data...")
df = pd.read_csv('data/processed/train_merged.csv', low_memory=False)
print(f"  ✓ {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── CATEGORIZE MISSING ─────────────────────────────────────
print("\n[2/5] Analyzing missing data...")
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100)
missing_df  = pd.DataFrame({'count': missing, 'pct': missing_pct}).query('count > 0')

drop_cols  = missing_df[missing_df['pct'] > 90].index.tolist()
high_miss  = missing_df[(missing_df['pct'] >= 50) & (missing_df['pct'] <= 90)].index.tolist()
med_miss   = missing_df[(missing_df['pct'] >= 10) & (missing_df['pct'] < 50)].index.tolist()
low_miss   = missing_df[(missing_df['pct'] > 0)  & (missing_df['pct'] < 10)].index.tolist()

print(f"  > 90% missing (DROP)          : {len(drop_cols)} columns")
print(f"  50-90% missing (FLAG+MEDIAN)  : {len(high_miss)} columns")
print(f"  10-50% missing (MEDIAN/MODE)  : {len(med_miss)} columns")
print(f"  < 10% missing (SIMPLE)        : {len(low_miss)} columns")
print(f"  No missing                    : {(missing_pct == 0).sum()} columns")

# ── APPLY STRATEGY ─────────────────────────────────────────
print("\n[3/5] Applying imputation strategy...")
df_clean = df.copy()

# Step 1: Drop >90%
df_clean.drop(columns=drop_cols, inplace=True)
print(f"  Step 1 — Dropped {len(drop_cols)} columns. Shape: {df_clean.shape}")

# Step 2: FLAG + MEDIAN for 50-90%
for col in high_miss:
    if col in df_clean.columns:
        df_clean[f'{col}_missing_flag'] = df_clean[col].isnull().astype(int)
        fill = df_clean[col].mode()[0] if df_clean[col].dtype == 'object' else df_clean[col].median()
        df_clean[col].fillna(fill, inplace=True)
print(f"  Step 2 — Added {len(high_miss)} missing flags. Shape: {df_clean.shape}")

# Step 3: MEDIAN/MODE for 10-50%
for col in med_miss:
    if col in df_clean.columns:
        fill = df_clean[col].mode()[0] if df_clean[col].dtype == 'object' else df_clean[col].median()
        df_clean[col].fillna(fill, inplace=True)
print(f"  Step 3 — Imputed {len(med_miss)} medium-missing columns.")

# Step 4: Simple fill for <10%
for col in low_miss:
    if col in df_clean.columns:
        fill = df_clean[col].mode()[0] if df_clean[col].dtype == 'object' else df_clean[col].median()
        df_clean[col].fillna(fill, inplace=True)
print(f"  Step 4 — Imputed {len(low_miss)} low-missing columns.")

# ── FIX REMAINING ──────────────────────────────────────────
print("\n[4/5] Fixing any remaining missing values...")
still = df_clean.isnull().sum()
still = still[still > 0]
for col in still.index:
    if df_clean[col].dtype == 'object':
        df_clean[col].fillna('Unknown', inplace=True)
    else:
        fv = df_clean[col].median()
        df_clean[col].fillna(0 if np.isnan(fv) else fv, inplace=True)

remaining = df_clean.isnull().sum().sum()
print(f"  Missing values remaining: {remaining}")

# ── ASSERT & SAVE ──────────────────────────────────────────
print("\n[5/5] Validating and saving...")
assert df_clean['isFraud'].sum() == df['isFraud'].sum(), "ERROR: Fraud cases lost!"
assert df_clean.shape[0] == df.shape[0], "ERROR: Rows lost!"
assert remaining == 0, "ERROR: Still has missing values!"

df_clean.to_csv('data/processed/train_clean.csv', index=False)

print(f"\n{'='*55}")
print(f"   NOTEBOOK 03 COMPLETE")
print(f"{'='*55}")
print(f"  Original shape : {df.shape[0]:,} x {df.shape[1]}")
print(f"  Clean shape    : {df_clean.shape[0]:,} x {df_clean.shape[1]}")
print(f"  Missing values : {remaining}")
print(f"  Fraud rate     : {df_clean['isFraud'].mean()*100:.3f}%")
print(f"  Rows lost      : 0")
print(f"  Saved          : data/processed/train_clean.csv ✓")
print(f"{'='*55}")
