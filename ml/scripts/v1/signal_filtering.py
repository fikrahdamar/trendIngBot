"""
    Theres a "Time Teleporter" bug  that scrambles data
    filter rows and then reset_index(drop=True), destroy the continuity of time.
    NOT USED (maybe)

"""

import pandas as pd

def filter_signals(
    df: pd.DataFrame,
    min_atr_pct=0.001,
    min_vol_ratio=1.2,
    max_price_ema_dist=0.02
) -> pd.DataFrame:

    original_len = len(df)
    df = df.dropna()

    df = df[df["atr_pct"] >= min_atr_pct]
    df = df[df["vol_ratio"] >= min_vol_ratio]

    df = df[
        (df["body_pct"] >= 0) &
        (df["upper_wick_pct"] >= 0) &
        (df["lower_wick_pct"] >= 0)
    ]


    if "price_ema20_dist" in df.columns:
        df = df[df["price_ema20_dist"].abs() <= max_price_ema_dist]

    filtered_len = len(df)

    print(f"[SignalFilter] Rows before: {original_len}")
    print(f"[SignalFilter] Rows after : {filtered_len}")
    print(f"[SignalFilter] Removed    : {original_len - filtered_len}")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df_btc = pd.read_csv("ml/data/processed/v1/BTCUSDT_15m_features.csv")
    btc_df_filtered = filter_signals(df_btc)
    btc_df_filtered.to_csv("ml/data/processed/v1/signal_filtering/BTCUSDT_15m_features_signal_filtering.csv", index=False)

    df_eth = pd.read_csv("ml/data/processed/v1/ETHUSDT_15m_features.csv")
    eth_df_filtered = filter_signals(df_eth)
    eth_df_filtered.to_csv("ml/data/processed/v1/signal_filtering/ETHUSDT_15m_features_signal_filtering.csv", index=False)