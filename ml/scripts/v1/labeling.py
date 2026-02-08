import pandas as pd
import numpy as np


def label_trades_atr(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_atr_mult: float = 4.0,
):
    """
    Labels trades using ATR-based TP/SL and ATR-based horizon.

    Labels:
        1 = GOOD TRADE (TP hit before SL in trend direction)
        0 = BAD / NO TRADE
    """

    df = df.copy().reset_index(drop=True)

    labels = []
    time_to_hit = []
    mfe_list = []
    mae_list = []

    for i in range(len(df)):
        trend = df.loc[i, "ema_trend"]

        if trend == 1:
            direction = "long"
        elif trend == -1:
            direction = "short"
        else:
            labels.append(0)
            time_to_hit.append(np.nan)
            mfe_list.append(0)
            mae_list.append(0)
            continue

        entry = df.loc[i, "close"]
        atr = df.loc[i, atr_col]

        if np.isnan(atr) or atr <= 0:
            labels.append(0)
            time_to_hit.append(np.nan)
            mfe_list.append(0)
            mae_list.append(0)
            continue

        if direction == "long":
            tp = entry + tp_mult * atr
            sl = entry - sl_mult * atr
        else:
            tp = entry - tp_mult * atr
            sl = entry + sl_mult * atr

        max_bars = int(max_atr_mult * 10)  
        max_bars = max(10, max_bars)  

        label = 0
        mfe = 0.0
        mae = 0.0
        resolved_at = None

        for j in range(1, min(max_bars + 1, len(df) - i)):
            high = df.loc[i + j, "high"]
            low = df.loc[i + j, "low"]

            if direction == "long":
                mfe = max(mfe, high - entry)
                mae = min(mae, low - entry)

                if high >= tp:
                    label = 1
                    resolved_at = j
                    break
                if low <= sl:
                    label = 0
                    resolved_at = j
                    break

            else:  # short
                mfe = max(mfe, entry - low)
                mae = min(mae, entry - high)

                if low <= tp:
                    label = 1
                    resolved_at = j
                    break
                if high >= sl:
                    label = 0
                    resolved_at = j
                    break

        labels.append(label)
        time_to_hit.append(resolved_at)
        mfe_list.append(mfe)
        mae_list.append(mae)

    df["label"] = labels
    df["time_to_hit"] = time_to_hit
    df["mfe"] = mfe_list
    df["mae"] = mae_list

    return df

if __name__ == "__main__":
    INPUT_PATH = "ml/data/processed/v1/signal_filtering/BTCUSDT_15m_features_signal_filtering.csv"
    OUTPUT_PATH = "ml/data/processed/v1/labeled/BTCUSDT_15m_labeled(2).csv"

    df = pd.read_csv(INPUT_PATH)
    labeled_df = label_trades_atr(df)

    print("Label distribution:")
    print(labeled_df["label"].value_counts(normalize=True))

    labeled_df.to_csv(OUTPUT_PATH, index=False)
    
    numeric_cols = labeled_df.select_dtypes(include=[np.number]).columns
    bad_cols = []

    for col in numeric_cols:
        nan_pct = labeled_df[col].isna().mean()
        inf_pct = np.isinf(labeled_df[col]).mean()

        if nan_pct > 0 or inf_pct > 0:
            bad_cols.append((col, nan_pct, inf_pct))

    print("Bad columns (NaN / inf):", bad_cols)
    
    print(labeled_df[[
        "rsi_14",
        "atr_pct",
        "vol_ratio",
        "price_ema20_dist"
    ]].describe())
    print(
    labeled_df
    .select_dtypes(include=[np.number])
    .var()
    .sort_values()
    .head(10)
)
    
    shifted = labeled_df.copy()
    feature_cols = [
    c for c in shifted.columns
    if c not in ["label", "time_to_hit", "mfe", "mae", "timestamp", "ema_trend"]
]

    shifted[feature_cols] = shifted[feature_cols].shift(1)

    
    corr_now = labeled_df[feature_cols].corrwith(labeled_df["label"]).abs().mean()
    corr_shift = shifted[feature_cols].corrwith(shifted["label"]).abs().mean()

    print("Corr now vs shifted:", corr_now, corr_shift)
