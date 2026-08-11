# ============================================================
# 11_export_demo_samples.py
# Run this ONCE, locally, where you have data/processed/train_final.csv
# It saves a small file of REAL transactions (with true labels) that
# the Streamlit app samples from for the "Score a Transaction" tab.
#
# Run: python notebooks/11_export_demo_samples.py
# (run from the project root, e.g. C:\Users\chava\payguard\)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

N_SAMPLES = 150          # how many rows to save for the demo
RANDOM_STATE = 42

print("Loading full training data (this is the big file, not committed to git)...")
df = pd.read_csv('data/processed/train_final.csv')

# Match the exact split used in notebook 10, so these are genuinely
# held-out rows the model has not seen.
y = df['isFraud']
X = df.drop(columns=['isFraud'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Make sure the model can actually score these columns.
model = joblib.load('models/xgb_tuned.pkl')
feature_names = model.get_booster().feature_names
missing = set(feature_names) - set(X_test.columns)
if missing:
    raise ValueError(f"X_test is missing columns the model expects: {missing}")
X_test = X_test[feature_names]

# Sample roughly balanced-ish: real fraud rate is ~3.5%, so a pure random
# sample of 150 would give ~5 fraud rows. Oversample fraud a bit so the
# demo has enough interesting cases to show off, but keep it majority-legit
# so it stays representative.
fraud_idx = y_test[y_test == 1].index
legit_idx = y_test[y_test == 0].index

n_fraud = min(len(fraud_idx), 30)              # up to 30 fraud examples
n_legit = N_SAMPLES - n_fraud

rng = np.random.RandomState(RANDOM_STATE)
sampled_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
sampled_legit = rng.choice(legit_idx, size=n_legit, replace=False)
sample_idx = np.concatenate([sampled_fraud, sampled_legit])
rng.shuffle(sample_idx)

demo_df = X_test.loc[sample_idx].copy()
demo_df['true_label'] = y_test.loc[sample_idx].values
demo_df = demo_df.reset_index(drop=True)

out_path = 'data/processed/demo_samples.csv'
demo_df.to_csv(out_path, index=False)

print(f"Saved {len(demo_df)} rows ({demo_df['true_label'].sum()} fraud, "
      f"{(demo_df['true_label']==0).sum()} legit) to {out_path}")
print(f"File size: {demo_df.memory_usage(deep=True).sum() / 1024:.1f} KB in memory")
print()
print("Next steps:")
print("1. Open .gitignore and add an exception so this one file IS tracked, e.g.:")
print("   data/processed/*")
print("   !data/processed/demo_samples.csv")
print("   !data/processed/threshold_config.csv")
print("   !data/processed/final_metrics.csv")
print("   !data/processed/imbalance_config.csv")
print("2. git add data/processed/demo_samples.csv .gitignore")
print("3. git commit -m 'Add demo sample transactions for live scoring tab'")
print("4. git push")
