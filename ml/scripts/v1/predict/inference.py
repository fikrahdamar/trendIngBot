import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json

class CryptoPredictor:
    def __init__(self, model_path, threshold_path):
        self.features = [
            "return_1", "return_3", "adx", "bb_width", "vol_zscore", 
            "rsi_14", "rsi_slope", "ema20_slope", "ema50_slope",
            "price_ema20_dist", "price_ema50_dist", "atr_pct", "vol_ratio",
            "body_pct", "upper_wick_pct", "lower_wick_pct", "ema_trend",
            "ema_trend_1h", "adx_1h", "rsi_14_1h", "ema20_slope_1h" 
        ]
        
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                self.threshold = float(f.read())
        else:
            self.threshold = 0.5
            
        self.exchange = ccxt.binance()
        
    def _calculate_adx(self, df, period=14):
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
        return df['DX'].ewm(alpha=alpha, adjust=False).mean()

    def _prepare_features(self, df_15m):
        df_1h = df_15m.set_index("timestamp").resample("1h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        df_1h["ema20"] = df_1h["close"].ewm(span=20).mean()
        df_1h["ema50"] = df_1h["close"].ewm(span=50).mean()
        df_1h["ema_trend"] = np.where(df_1h["ema20"] > df_1h["ema50"], 1, -1)
        df_1h["ema20_slope"] = df_1h["ema20"].pct_change() * 1000
        
        delta = df_1h["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df_1h["rsi_14"] = 100 - (100 / (1 + rs))
        df_1h["adx"] = self._calculate_adx(df_1h)
        
        cols_to_keep = ["ema_trend", "rsi_14", "adx", "ema20_slope"]
        df_1h = df_1h[cols_to_keep].add_suffix("_1h")
        df_1h = df_1h.shift(1) 

        df = df_15m.merge(df_1h, on="timestamp", how="left").ffill()
        
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_slope"] = df["rsi_14"].diff(3)
        
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        df["ema20_slope"] = df["ema20"].pct_change() * 1000
        df["ema50_slope"] = df["ema50"].pct_change() * 1000
        df["ema_trend"] = np.where(df["ema20"] > df["ema50"], 1, -1)
        
        df["price_ema20_dist"] = (df["close"] - df["ema20"]) / df["close"]
        df["price_ema50_dist"] = (df["close"] - df["ema50"]) / df["close"]
        
        df["adx"] = self._calculate_adx(df)
        bb_std = df["close"].rolling(20).std()
        df["bb_width"] = (4 * bb_std) / df["ema20"]
        
        high_low = df["high"] - df["low"]
        tr = pd.concat([high_low, np.abs(df["high"] - df["close"].shift()), np.abs(df["low"] - df["close"].shift())], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr_14"] / df["close"]
  
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma20"]
        df["vol_zscore"] = (df["volume"] - df["vol_sma20"]) / (df["volume"].rolling(20).std() + 1e-9)
        
        df["return_1"] = df["close"].pct_change()
        df["return_3"] = df["close"].pct_change(3)
        
        body = np.abs(df["close"] - df["open"])
        range_ = (df["high"] - df["low"]).replace(0, 1e-9)
        df["body_pct"] = body / range_
        df["upper_wick_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / range_
        df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / range_
   
        return df.iloc[-1:] 
    
    def analyze(self, symbol="BTC/USDT"):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe="15m", limit=500)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            latest_features = self._prepare_features(df)
            
            dmatrix = xgb.DMatrix(latest_features[self.features])
            probability = self.model.predict(dmatrix)[0]
            
            current_price = latest_features["close"].values[0]
            atr = latest_features["atr_14"].values[0]
            rsi = latest_features["rsi_14"].values[0]
            
            sentiment = "NO TRADE"
            if probability > self.threshold:
                if rsi < 45: sentiment = "LONG"
                elif rsi > 55: sentiment = "SHORT"
                
            tp = 0
            sl = 0
            if sentiment == "LONG":
                tp = current_price + (1.5 * atr)
                sl = current_price - (1.0 * atr)
            elif sentiment == "SHORT":
                tp = current_price - (1.5 * atr)
                sl = current_price + (1.0 * atr)
            
            return {
                "symbol": symbol,
                "price": current_price,
                "recommendation": sentiment,
                "confidence": round(float(probability), 4),
                "threshold_used": round(self.threshold, 4),
                "entry_zone": f"{current_price}",
                "tp": round(tp, 2),
                "sl": round(sl, 2),
                "features": {
                    "rsi": round(rsi, 2),
                    "vol_ratio": round(latest_features["vol_ratio"].values[0], 2),
                    "adx": round(latest_features["adx"].values[0], 2)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    bot = CryptoPredictor("ml/models/v1/reversion_model_v1.json", "ml/models/v1/threshold.txt")
    result = bot.analyze("BTC/USDT")
    print(json.dumps(result, indent=4))