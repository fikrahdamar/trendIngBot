import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

# 1. LOAD DATA
DATA_PATH = "ml/data/processed/v1/labeled/BTCUSDT_15m_labeled(FULL).csv"

# 2. DEFINE FEATURES
FEATURES = [
    "return_1", "return_3", "adx", "bb_width",
    "vol_zscore", "hour",
    "rsi_14", "rsi_slope",
    "ema20_slope", "ema50_slope",
    "price_ema20_dist", "price_ema50_dist",
    "atr_pct", "vol_ratio",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "ema_trend",
    
    # --- NEW MTF FEATURES ---
    "ema_trend_1h", "adx_1h", "rsi_14_1h", "ema20_slope_1h" 
]

TARGET = "label"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# 3. CREATE LAGS
feature_cols = [f for f in FEATURES if f != "ema_trend"]
for f in feature_cols:
    df[f"{f}_lag1"] = df[f].shift(1)

# 4. FILTERING ("Sniper Mode")
mask = (
    (df["adx_lag1"] > 20) &                # 15m Trend Strength OK
    (df["adx_1h_lag1"] > 20) &             # 1H Trend Strength OK (NEW!)
    (df["vol_ratio_lag1"] > 1.0) &         
    (df["label"].isin([0, 1])) &           
    (df["price_ema20_dist_lag1"].abs() < 0.015) & 
    
    # *** THE MAGIC FILTER ***
    # Only trade if 15m Trend matches 1H Trend
    (df["ema_trend"] == df["ema_trend_1h"]) 
)

df_model = df[mask].copy().reset_index(drop=True)

print(f"Original rows: {len(df)}")
print(f"Sniper setups found: {len(df_model)}")

# 5. PREPARE X, y
X = df_model[[f"{f}_lag1" for f in feature_cols]].copy()
X["ema_trend"] = df_model["ema_trend"].values
y = df_model["label"]

# 6. TIME-SERIES SPLIT
tscv = TimeSeriesSplit(n_splits=5)

# UNLOCKING THE MODEL
model = xgb.XGBClassifier(
    n_estimators=1000,       
    learning_rate=0.01,      
    max_depth=5,             # Increased depth slightly
    subsample=0.8,           
    colsample_bytree=0.8,    
    objective='binary:logistic',
    scale_pos_weight=1.5,    # GENTLE NUDGE: Encourage it to trade, but don't force it
    gamma=0.1,               # Minimum loss reduction required to make a further partition
    reg_lambda=1.0,          # L2 regularization (Standard)
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50 
)

for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

print(f"\nFinal Train size: {len(X_train)} | Val size: {len(X_val)}")

# 7. TRAIN
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100
)

# 8. EVALUATE
# We lower the threshold to 0.45 because the model is naturally conservative
probs = model.predict_proba(X_val)[:, 1]
preds = (probs >= 0.45).astype(int) 

train_probs = model.predict_proba(X_train)[:, 1]
train_roc = roc_auc_score(y_train, train_probs)
val_roc = roc_auc_score(y_val, probs)

print(f"\nTrain ROC-AUC: {round(train_roc, 4)}")
print(f"Val ROC-AUC:   {round(val_roc, 4)}")

print("\nClassification report (0.45 threshold):")
print(classification_report(y_val, preds, digits=4))

print("Confusion matrix:")
print(confusion_matrix(y_val, preds))

# Feature Importance
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Predictors (XGBoost):")
print(importances.head(10))