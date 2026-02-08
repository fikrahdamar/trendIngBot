import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)


DATA_PATH = "ml/data/processed/v1/labeled/BTCUSDT_15m_labeled(2).csv"

FEATURES = [
    "return_1",
    "return_3",
    "rsi_14",
    "rsi_slope",
    "ema20_slope",
    "ema50_slope",
    "price_ema20_dist",
    "price_ema50_dist",
    "atr_pct",
    "vol_ratio",
    "vol_spike",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "ema_trend",   # allowed
]

TARGET = "label"
TRAIN_RATIO = 0.7

df = pd.read_csv(DATA_PATH)

df = df[df["ema_trend"] != 0].reset_index(drop=True)

X = df[FEATURES]
y = df[TARGET]


split_idx = int(len(df) * TRAIN_RATIO)

X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
print(f"Positive rate (train): {y_train.mean():.3f}")
print(f"Positive rate (val):   {y_val.mean():.3f}")


model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1
    ))
])

model.fit(X_train, y_train)


probs = model.predict_proba(X_val)[:, 1]
preds = (probs >= 0.5).astype(int)

roc = roc_auc_score(y_val, probs)

print("\nROC-AUC:", round(roc, 4))
print("\nClassification report (0.5 threshold):")
print(classification_report(y_val, preds, digits=4))

print("Confusion matrix:")
print(confusion_matrix(y_val, preds))


def precision_at_k(y_true, y_prob, k):
    k = int(len(y_prob) * k)
    idx = np.argsort(y_prob)[-k:]
    return y_true.iloc[idx].mean()

print("\nPrecision @ confidence buckets:")
for k in [0.1, 0.2, 0.3]:
    print(f" Top {int(k*100)}%: {precision_at_k(y_val, probs, k):.3f}")


coefs = pd.Series(
    model.named_steps["clf"].coef_[0],
    index=FEATURES
).sort_values(key=np.abs, ascending=False)

print("\nTop feature weights:")
print(coefs.head(10))
