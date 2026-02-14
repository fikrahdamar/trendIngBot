import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TIMEFRAME = "15m"
YEARS_BACK = 5
EXCHANGE_NAME = "binance"

SAVE_DIR = "ml/data/raw/v1/"
LIMIT = 1000  


def ms_since(date):
    return int(date.timestamp() * 1000)


def fetch_ohlcv(exchange, symbol, timeframe, since, limit=1000):
    all_candles = []

    while True:
        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

        if not candles:
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 1

        # prevent rate limit issues
        time.sleep(exchange.rateLimit / 1000)

        if len(candles) < limit:
            break

    return all_candles


def main():
    exchange = getattr(ccxt, EXCHANGE_NAME)({
        "enableRateLimit": True,
    })

    os.makedirs(SAVE_DIR, exist_ok=True)

    since_date = datetime.utcnow() - timedelta(days=365 * YEARS_BACK)
    since_ms = ms_since(since_date)

    for symbol in SYMBOLS:
        print(f"Fetching {symbol} {TIMEFRAME} data...")

        candles = fetch_ohlcv(
            exchange,
            symbol,
            TIMEFRAME,
            since_ms,
            LIMIT
        )

        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        filename = f"{symbol.replace('/', '')}_{TIMEFRAME}(5Y).csv"
        path = os.path.join(SAVE_DIR, filename)

        df.to_csv(path, index=False)
        print(f"Saved → {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
