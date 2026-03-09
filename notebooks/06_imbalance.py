import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import joblib

os.chdir(r"C:\Users\chava\payguard")
print("NOTEBOOK 06 - Imbalance Strategy")
df = pd.read_csv("data/processed/train_final.csv", nrows=80000)
df["isFraud"] = df["isFraud"].astype(int)
X = df.drop("isFraud", axis=1).fillna(0)
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)
spw = (y_train==0).sum() / (y_train==1).sum()
print("SPW: " + str(round(spw,2)))
lr = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
lr.fit(Xtr, y_train)
score = average_precision_score(y_test, lr.predict_proba(Xte)[:,1])
print("PR-AUC: " + str(round(score,4)))
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
pd.DataFrame({"parameter":["scale_pos_weight"],"value":[str(round(spw,2))]}).to_csv("data/processed/imbalance_config.csv", index=False)
print("NOTEBOOK 06 COMPLETE")
