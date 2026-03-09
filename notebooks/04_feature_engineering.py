# ============================================================
# PAYGUARD — NOTEBOOK 04: Feature Engineering
# Run from: C:\Users\chava\payguard\
# Command : python notebooks\04_feature_engineering.py
# ============================================================

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\chava\payguard'
os.chdir(ROOT)

print("=" * 55)
print("   NOTEBOOK 04 — Feature Engineering")
print("=" * 55)

# ── LOAD ───────────────────────────────────────────────────
print("\n[1/7] Loading clean data...")
df = pd.read_csv('data/processed/train_clean.csv', low_memory=False)
print(f"  ✓ {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── TIME FEATURES ──────────────────────────────────────────
print("\n[2/7] Time features...")
df['hour']             = (df['TransactionDT'] // 3600) % 24
df['day_of_week']      = (df['TransactionDT'] // (3600*24)) % 7
df['day_of_month']     = (df['TransactionDT'] // (3600*24)) % 30
df['week_of_year']     = (df['TransactionDT'] // (3600*24*7)) % 52
df['is_night']         = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)
df['is_peak_fraud_hour'] = (df['hour'] == 7).astype(int)
df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
df['is_night_weekend'] = ((df['is_night']==1) & (df['is_weekend']==1)).astype(int)
print(f"  ✓ is_peak_fraud_hour=1: {df[df['is_peak_fraud_hour']==1]['isFraud'].mean()*100:.2f}% fraud")

# ── AMOUNT FEATURES ────────────────────────────────────────
print("\n[3/7] Amount features...")
df['amt_log']          = np.log1p(df['TransactionAmt'])
df['is_micro_txn']     = (df['TransactionAmt'] < 10).astype(int)
df['is_high_value']    = (df['TransactionAmt'] > 500).astype(int)
df['is_round_amount']  = (df['TransactionAmt'] % 1 == 0).astype(int)

card_stats = df.groupby('card1')['TransactionAmt'].agg(['mean','std','median'])
card_stats.columns = ['card_amt_mean','card_amt_std','card_amt_median']
df = df.merge(card_stats, on='card1', how='left')
df['amt_zscore']          = (df['TransactionAmt'] - df['card_amt_mean']) / (df['card_amt_std'] + 1e-8)
df['amt_to_median_ratio'] = df['TransactionAmt'] / (df['card_amt_median'] + 1e-8)
print(f"  ✓ is_micro_txn=1: {df[df['is_micro_txn']==1]['isFraud'].mean()*100:.2f}% fraud")
print(f"  ✓ amt_zscore fraud={df[df['isFraud']==1]['amt_zscore'].mean():.4f} legit={df[df['isFraud']==0]['amt_zscore'].mean():.4f}")

# ── EMAIL FEATURES ─────────────────────────────────────────
print("\n[4/7] Email features...")
high_risk_emails = ['mail.com','outlook.es','aim.com','hotmail.es',
                    'live.com.mx','anonymous.com','outlook.com']

df['p_email_high_risk']  = df['P_emaildomain'].isin(high_risk_emails).astype(int)
df['r_email_high_risk']  = df['R_emaildomain'].isin(high_risk_emails).astype(int)

def is_foreign(domain):
    if pd.isna(domain): return 0
    tld = str(domain).split('.')[-1].lower()
    return 0 if tld in ['com','net','org','edu','gov'] else 1

df['p_email_foreign']    = df['P_emaildomain'].apply(is_foreign)
df['r_email_foreign']    = df['R_emaildomain'].apply(is_foreign)
df['email_domain_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)

p_efr = df.groupby('P_emaildomain')['isFraud'].mean()
r_efr = df.groupby('R_emaildomain')['isFraud'].mean()
df['p_email_fraud_rate'] = df['P_emaildomain'].map(p_efr).fillna(df['isFraud'].mean())
df['r_email_fraud_rate'] = df['R_emaildomain'].map(r_efr).fillna(df['isFraud'].mean())
print(f"  ✓ p_email_high_risk=1: {df[df['p_email_high_risk']==1]['isFraud'].mean()*100:.2f}% fraud")

# ── DEVICE FEATURES ────────────────────────────────────────
print("\n[5/7] Device features...")
df['is_mobile']  = (df['DeviceType'] == 'mobile').astype(int)
df['is_desktop'] = (df['DeviceType'] == 'desktop').astype(int)
df['has_device'] = df['DeviceType'].notna().astype(int)

def get_os(d):
    if pd.isna(d): return 'unknown'
    d = str(d).lower()
    if 'windows' in d: return 'windows'
    if 'ios' in d or 'iphone' in d or 'ipad' in d: return 'ios'
    if 'android' in d: return 'android'
    if 'mac' in d: return 'mac'
    return 'other'

df['device_os']            = df['DeviceInfo'].apply(get_os)
os_fr                      = df.groupby('device_os')['isFraud'].mean()
df['device_os_fraud_rate'] = df['device_os'].map(os_fr)
print(f"  ✓ is_mobile=1: {df[df['is_mobile']==1]['isFraud'].mean()*100:.2f}% fraud")

# ── CARD + ADDRESS FEATURES ────────────────────────────────
print("\n[6/7] Card & address features...")
for col in ['card4','card6','card3']:
    fr = df.groupby(col)['isFraud'].mean()
    df[f'{col}_fraud_rate'] = df[col].map(fr).fillna(df['isFraud'].mean())

df['card1_txn_count']    = df['card1'].map(df.groupby('card1')['TransactionID'].count())
df['card1_card2']        = df['card1'].astype(str) + '_' + df['card2'].astype(str)
df['card_combo_count']   = df['card1_card2'].map(df.groupby('card1_card2')['TransactionID'].count())
df['card_email_nunique'] = df['card1'].map(df.groupby('card1')['P_emaildomain'].nunique())
df['card_multi_email']   = (df['card_email_nunique'] > 1).astype(int)

df['addr1_frequency']    = df['addr1'].map(df.groupby('addr1')['TransactionID'].count()).fillna(0)
df['addr1_rare']         = (df['addr1_frequency'] < 10).astype(int)
df['is_international']   = (df['addr2'] != 87).astype(int)
print(f"  ✓ card_multi_email=1  : {df[df['card_multi_email']==1]['isFraud'].mean()*100:.2f}% fraud")
print(f"  ✓ is_international=1  : {df[df['is_international']==1]['isFraud'].mean()*100:.2f}% fraud")

# ── SAVE ───────────────────────────────────────────────────
print("\n[7/7] Saving engineered dataset...")
df.to_csv('data/processed/train_engineered.csv', index=False)

new_feats = [
    'hour','day_of_week','day_of_month','week_of_year','is_night','is_peak_fraud_hour',
    'is_weekend','is_night_weekend','amt_log','is_micro_txn','is_high_value',
    'is_round_amount','card_amt_mean','card_amt_std','card_amt_median','amt_zscore',
    'amt_to_median_ratio','p_email_high_risk','r_email_high_risk','p_email_foreign',
    'r_email_foreign','email_domain_match','p_email_fraud_rate','r_email_fraud_rate',
    'is_mobile','is_desktop','has_device','device_os','device_os_fraud_rate',
    'card4_fraud_rate','card6_fraud_rate','card3_fraud_rate','card1_txn_count',
    'card_combo_count','card_email_nunique','card_multi_email',
    'addr1_frequency','addr1_rare','is_international'
]

print(f"\n{'='*55}")
print(f"   NOTEBOOK 04 COMPLETE")
print(f"{'='*55}")
print(f"  New features   : {len(new_feats)}")
print(f"  Final shape    : {df.shape[0]:,} x {df.shape[1]}")
print(f"  Fraud rate     : {df['isFraud'].mean()*100:.3f}%")
print(f"  Saved          : data/processed/train_engineered.csv ✓")
print(f"{'='*55}")
