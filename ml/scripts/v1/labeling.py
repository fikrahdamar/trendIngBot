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
        1  = LONG (TP hit first)
       -1  = SHORT (TP hit first)
        0  = NO TRADE (SL first or no hit)
    """

    df = df.copy().reset_index(drop=True)

    labels = []
    time_to_hit = []
    mfe_list = []
    mae_list = []

    for i in range(len(df)):
        entry = df.loc[i, "close"]
        atr = df.loc[i, atr_col]

        if np.isnan(atr) or atr <= 0:
            labels.append(0)
            time_to_hit.append(np.nan)
            mfe_list.append(0)
            mae_list.append(0)
            continue

        tp_long = entry + tp_mult * atr
        sl_long = entry - sl_mult * atr
        tp_short = entry - tp_mult * atr
        sl_short = entry + sl_mult * atr

        max_bars = int(max_atr_mult * atr / atr)  
        max_bars = max(5, max_bars)  

        label = 0
        mfe = 0.0
        mae = 0.0
        resolved_at = None

        for j in range(1, min(max_bars + 1, len(df) - i)):
            high = df.loc[i + j, "high"]
            low = df.loc[i + j, "low"]

          
            mfe = max(mfe, high - entry)
            mae = min(mae, low - entry)

       
            if high >= tp_long:
                label = 1
                resolved_at = j
                break
            if low <= sl_long:
                label = 0
                resolved_at = j
                break

       
            if low <= tp_short:
                label = -1
                resolved_at = j
                break
            if high >= sl_short:
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
    OUTPUT_PATH = "ml/data/processed/v1/labeled/BTCUSDT_15m_labeled.csv"

    df = pd.read_csv(INPUT_PATH)
    labeled_df = label_trades_atr(df)

    print("Label distribution:")
    print(labeled_df["label"].value_counts(normalize=True))

    labeled_df.to_csv(OUTPUT_PATH, index=False)
