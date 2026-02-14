import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = "ml/data/processed/v1/labeled/reversion/SOLUSDT_15m_reversion(5Y).csv"

FEATURES = [
    "return_1", "return_3", "adx", "bb_width",
    "vol_zscore", 
    "rsi_14", "rsi_slope",
    "ema20_slope", "ema50_slope",
    "price_ema20_dist", "price_ema50_dist",
    "atr_pct", "vol_ratio",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "ema_trend",
    
    # MTF Features
    "ema_trend_1h", "adx_1h", "rsi_14_1h", "ema20_slope_1h" 
]

TARGET = "label"
df = pd.read_csv(DATA_PATH)
X_raw = df[FEATURES].copy()

mask = (
    # Price must be strictly oversold/overbought
    ((df["rsi_14"] < 30) | (df["rsi_14"] > 70)) &
    # don't trade dead markets
    (df["bb_width"] > 0.005) & 
    (df["label"].isin([0, 1]))
)

df_model = df[mask].copy().reset_index(drop=True)
X = df_model[FEATURES].copy()
y = df_model["label"]

print("SOLUSDT")
print(f"Original rows: {len(df)}")
print(f"Strict Reversion setups found: {len(df_model)}")

tscv = TimeSeriesSplit(n_splits=5)

model = xgb.XGBClassifier(
    n_estimators=500,        
    learning_rate=0.02,      
    max_depth=3,            
    subsample=0.7,           
    colsample_bytree=0.7,    
    objective='binary:logistic',
    scale_pos_weight=1.0,   
    gamma=0.2,              
    reg_lambda=1.0,          
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50 
)

for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

print(f"\nFinal Train size: {len(X_train)} | Val size: {len(X_val)}")
print(f"Positive Rate (Val): {y_val.mean():.3f}")

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=0 
)

probs = model.predict_proba(X_val)[:, 1]
print(f"Max Probability Score: {probs.max():.4f}")
print(f"Avg Probability Score: {probs.mean():.4f}")
print(f"Top 10 Scores: {np.sort(probs)[-10:]}")


dynamic_threshold = np.percentile(probs, 95) 
print(f"\nAPPLYING Dynamic Threshold (Top 5%): {dynamic_threshold:.4f}")

preds = (probs >= dynamic_threshold).astype(int)

train_probs = model.predict_proba(X_train)[:, 1]
train_roc = roc_auc_score(y_train, train_probs)
val_roc = roc_auc_score(y_val, probs)

print(f"\nTrain ROC-AUC: {round(train_roc, 4)}")
print(f"Val ROC-AUC:   {round(val_roc, 4)}")

print("\nClassification report (dynamic threshold):")
print(classification_report(y_val, preds, digits=4))

print("Confusion matrix:")
print(confusion_matrix(y_val, preds))

importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Predictors:")
print(importances.head(10))