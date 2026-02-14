import pandas as pd
import numpy as np
import os

def calculate_adx(df, period=14):
    """ Computes ADX manually. """
    alpha = 1/period
    df = df.copy()
    
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = np.abs(df['high'] - df['close'].shift(1))
    df['L-C'] = np.abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    
    df['UpMove'] = df['high'] - df['high'].shift(1)
    df['DownMove'] = df['low'].shift(1) - df['low']
    
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    
    df['TR_smooth'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DM_smooth'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM_smooth'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    
    df['DX'] = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['adx'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    
    return df['adx']

def calculate_features(df):
    """ Calculates standard features for a given dataframe (any timeframe) """
    df = df.copy()
    
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    
    df["ema20_slope"] = df["ema20"].pct_change() * 1000
    df["ema50_slope"] = df["ema50"].pct_change() * 1000
    
    df["ema_trend"] = np.where(df["ema20"] > df["ema50"], 1, -1)
    
    df["adx"] = calculate_adx(df)
    
    df["bb_width"] = (df["close"].rolling(20).std() * 4) / df["ema20"]
    
    return df

def process_ohlcv(input_path: str, output_path: str):
    print(f"Processing {input_path}...")
    df_15m = pd.read_csv(input_path)
    df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
    df_15m = df_15m.sort_values("timestamp").reset_index(drop=True)

    df_1h = df_15m.set_index("timestamp").resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    df_1h = calculate_features(df_1h)
    
    cols_to_keep = ["ema_trend", "rsi_14", "adx", "ema20_slope"]
    df_1h = df_1h[cols_to_keep].add_suffix("_1h")
    
    df_1h = df_1h.shift(1)

    df_combined = df_15m.merge(df_1h, on="timestamp", how="left").ffill()

    df_combined = calculate_features(df_combined)
    
    df_combined["return_1"] = df_combined["close"].pct_change()
    df_combined["return_3"] = df_combined["close"].pct_change(3)
    df_combined["rsi_slope"] = df_combined["rsi_14"].diff(3)
    
    df_combined["price_ema20_dist"] = (df_combined["close"] - df_combined["ema20"]) / df_combined["close"]
    df_combined["price_ema50_dist"] = (df_combined["close"] - df_combined["ema50"]) / df_combined["close"]
    
    high_low = df_combined["high"] - df_combined["low"]
    tr = pd.concat([high_low, np.abs(df_combined["high"] - df_combined["close"].shift()), np.abs(df_combined["low"] - df_combined["close"].shift())], axis=1).max(axis=1)
    df_combined["atr_14"] = tr.rolling(14).mean()
    df_combined["atr_pct"] = df_combined["atr_14"] / df_combined["close"]

    df_combined["vol_sma20"] = df_combined["volume"].rolling(20).mean()
    df_combined["vol_ratio"] = df_combined["volume"] / df_combined["vol_sma20"]
    df_combined["vol_zscore"] = (df_combined["volume"] - df_combined["vol_sma20"]) / (df_combined["volume"].rolling(20).std() + 1e-9)
    
    body = np.abs(df_combined["close"] - df_combined["open"])
    range_ = (df_combined["high"] - df_combined["low"]).replace(0, 1e-9)
    df_combined["body_pct"] = body / range_
    df_combined["upper_wick_pct"] = (df_combined["high"] - df_combined[["open", "close"]].max(axis=1)) / range_
    df_combined["lower_wick_pct"] = (df_combined[["open", "close"]].min(axis=1) - df_combined["low"]) / range_

    df_combined["hour"] = df_combined["timestamp"].dt.hour
    
    # 1 = Strong Buy (15m Up + 1H Up)
    # -1 = Strong Sell (15m Down + 1H Down)
    # 0 = Conflict (Don't trade)
    df_combined["trend_alignment"] = np.where(
        df_combined["ema_trend"] == df_combined["ema_trend_1h"], 
        df_combined["ema_trend"], 
        0
    )

    df_combined = df_combined.dropna().reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"Saved {len(df_combined)} rows with MTF features to {output_path}")

if __name__ == "__main__":
    process_ohlcv("ml/data/raw/v1/BTCUSDT_15m(5Y).csv", "ml/data/processed/v1/BTCUSDT_15m_features(5Y).csv")
    process_ohlcv("ml/data/raw/v1/SOLUSDT_15m(5Y).csv", "ml/data/processed/v1/SOLUSDT_15m_features(5Y).csv")
    process_ohlcv("ml/data/raw/v1/ETHUSDT_15m(5Y).csv", "ml/data/processed/v1/ETHUSDT_15m_features(5Y).csv")
    process_ohlcv("ml/data/raw/v1/BNBUSDT_15m(5Y).csv", "ml/data/processed/v1/BNBUSDT_15m_features(5Y).csv")