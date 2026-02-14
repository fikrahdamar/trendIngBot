import pandas as pd
import numpy as np
import os

def label_mean_reversion(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    sl_mult: float = 1.0, 
    max_bars: int = 24    
):
    """
    Labels trades for MEAN REVERSION.
    Setup: Price closes OUTSIDE the Bollinger Bands.
    Target: Return to the Mean (EMA20).
    Stop: Continue moving away by 1 ATR.
    """
    df = df.copy().reset_index(drop=True)
    
    std_dev = df["close"].rolling(20).std()
    df["bb_upper"] = df["ema20"] + (2.0 * std_dev)
    df["bb_lower"] = df["ema20"] - (2.0 * std_dev)

    labels = []

    for i in range(len(df)):
        close = df.loc[i, "close"]
        lower_band = df.loc[i, "bb_lower"]
        upper_band = df.loc[i, "bb_upper"]
        ema20 = df.loc[i, "ema20"]
        atr = df.loc[i, atr_col]
        
        if np.isnan(atr) or atr <= 0 or np.isnan(ema20):
            labels.append(np.nan)
            continue

        # LOGIC:
        # 1. LONG REVERSION: Close < Lower Band
        # 2. SHORT REVERSION: Close > Upper Band
        
        trade_type = None
        if close < lower_band:
            trade_type = "long_reversion"
            tp = ema20 
            sl = close - (sl_mult * atr)
            
        elif close > upper_band:
            trade_type = "short_reversion"
            tp = ema20 
            sl = close + (sl_mult * atr)
            
        else:
            labels.append(np.nan) 
            continue

        label = 0 
        
        for j in range(1, min(max_bars + 1, len(df) - i)):
            future_high = df.loc[i + j, "high"]
            future_low = df.loc[i + j, "low"]
            future_ema = df.loc[i + j, "ema20"]
            
            dynamic_tp = future_ema

            if trade_type == "long_reversion":
                if future_low <= sl:
                    label = 0
                    break
                if future_high >= dynamic_tp:
                    label = 1
                    break
            
            elif trade_type == "short_reversion":
                if future_high >= sl:
                    label = 0
                    break
                if future_low <= dynamic_tp:
                    label = 1
                    break
        
        labels.append(label)

    df["label"] = labels
    return df

if __name__ == "__main__":
    INPUT_PATH = "ml/data/processed/v1/BNBUSDT_15m_features(5Y).csv"
    OUTPUT_PATH = "ml/data/processed/v1/labeled/reversion/BNBUSDT_15m_reversion(5Y).csv"

    if not os.path.exists(INPUT_PATH):
        print("Run feature_engineering.py first!")
        exit()

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    labeled_df = label_mean_reversion(df)
    
    labeled_df = labeled_df.dropna(subset=["label"])
    
    print("Label distribution (Reversion Strategy):")
    print(labeled_df["label"].value_counts(normalize=True))
    print(f"Total Reversion Setups: {len(labeled_df)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    labeled_df.to_csv(OUTPUT_PATH, index=False)