import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_historical_data(symbol, timeframe='15m', years=2):
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })

    since = exchange.parse8601((datetime.now() - timedelta(days=years*365)).isoformat())
    
    all_ohlcv = []
    print(f"Starting fetch for {symbol}...")

    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not ohlcv:
                break
            
            since = ohlcv[-1][0] + 1
            all_ohlcv += ohlcv
            
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000).strftime('%Y-%m-%d')
            print(f"Fetched up to {current_date} | Total rows: {len(all_ohlcv)}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    os.makedirs('ml/data/raw/v2/', exist_ok=True)
    filename = f"ml/data/raw/v2/{symbol.replace('/', '')}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    print(f"Successfully saved: {filename}")

if __name__ == "__main__":
    symbols = ['BTC/USDT', 'ETH/USDT']
    for sym in symbols:
        fetch_historical_data(sym, timeframe='15m', years=2)