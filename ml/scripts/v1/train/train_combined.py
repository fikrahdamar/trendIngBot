import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os


FILES = [
    "ml/data/processed/v1/labeled/reversion/BTCUSDT_15m_reversion(5Y).csv",
    "ml/data/processed/v1/labeled/reversion/BNBUSDT_15m_reversion(5Y).csv",
    "ml/data/processed/v1/labeled/reversion/ETHUSDT_15m_reversion(5Y).csv"
]

FEATURES = [
    "return_1", "return_3", "adx", "bb_width",
    "vol_zscore", 
    "rsi_14", "rsi_slope",
    "ema20_slope", "ema50_slope",
    "price_ema20_dist", "price_ema50_dist",
    "atr_pct", "vol_ratio",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "ema_trend",
    "ema_trend_1h", "adx_1h", "rsi_14_1h", "ema20_slope_1h" 
]

TARGET = "label"
MODEL_PATH = "ml/models/v1/reversion_model_v1.json"

dfs = []
for f in FILES:
    _df = pd.read_csv(f)
    _df["source_file"] = f 
    dfs.append(_df)

df = pd.concat(dfs, ignore_index=True)
print(f"Total Combined Rows: {len(df)}")
X_raw = df[FEATURES].copy()

mask = (
    ((df["rsi_14"] < 35) | (df["rsi_14"] > 65)) & 
    (df["bb_width"] > 0.005) & 
    (df["label"].isin([0, 1]))
)

df_model = df[mask].copy().reset_index(drop=True)
X = df_model[FEATURES].copy()
y = df_model["label"]

print(f"Setups found (Combined): {len(df_model)}")
tscv = TimeSeriesSplit(n_splits=5)

model = xgb.XGBClassifier(
    n_estimators=800,        
    learning_rate=0.015,      
    max_depth=4,             
    subsample=0.65,           
    colsample_bytree=0.65,    
    objective='binary:logistic',
    scale_pos_weight=1.0, 
    gamma=0.2,               
    reg_lambda=2.0,         
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50 
)

print("\nTraining Cross-Validation")
fold = 1
for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    probs = model.predict_proba(X_val)[:, 1]
    
    dyn_thresh = np.percentile(probs, 95)
    preds = (probs >= dyn_thresh).astype(int)
    
    precision = classification_report(y_val, preds, output_dict=True)["1.0"]["precision"]
    print(f"Fold {fold} - AUC: {roc_auc_score(y_val, probs):.4f} | Precision (Top 5%): {precision:.4f} | Thresh: {dyn_thresh:.4f}")
    fold += 1

print("\nTraining Final Model on ALL Data")
model.set_params(early_stopping_rounds=None)
model.fit(X, y, verbose=False)

os.makedirs("ml/models/v1/", exist_ok=True)
model.save_model(MODEL_PATH)

all_probs = model.predict_proba(X)[:, 1]
final_threshold = np.percentile(all_probs, 95)
print(f"FINAL PRODUCTION THRESHOLD: {final_threshold:.4f}")

with open("ml/models/v1/threshold.txt", "w") as f:
    f.write(str(final_threshold))