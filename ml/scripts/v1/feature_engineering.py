import pandas as pd
import numpy as np
import os


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def process_ohlcv(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Returns
    df["return_1"] = df["close"].pct_change()
    df["return_3"] = df["close"].pct_change(3)

    # RSI
    df["rsi_14"] = compute_rsi(df["close"])
    df["rsi_slope"] = df["rsi_14"].diff()

    # EMA
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    df["ema20_slope"] = df["ema20"].pct_change()
    df["ema50_slope"] = df["ema50"].pct_change()

    df["price_ema20_dist"] = (df["close"] - df["ema20"]) / df["close"]
    df["price_ema50_dist"] = (df["close"] - df["ema50"]) / df["close"]

    df["ema_trend"] = (df["ema20"] > df["ema50"]).astype(int)

    # ATR
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    # Volume
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"]
    df["vol_spike"] = (df["vol_ratio"] > 1.5).astype(int)

    # Candle structure
    body = np.abs(df["close"] - df["open"])
    range_ = df["high"] - df["low"]

    df["body_pct"] = body / range_
    df["upper_wick_pct"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / range_
    df["lower_wick_pct"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / range_


    df = df.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    process_ohlcv(
        "ml/data/raw/v1/BTCUSDT_15m.csv",
        "ml/data/processed/v1/BTCUSDT_15m_features.csv"
    )

    process_ohlcv(
        "ml/data/raw/v1/ETHUSDT_15m.csv",
        "ml/data/processed/v1/ETHUSDT_15m_features.csv"
    )