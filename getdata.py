import pandas as pd
from binance.client import Client
import time
import secerts

api_key = secerts.bin_api_key
api_secret = secerts.bin_api_secret

client = Client(api_key, api_secret, tld='us')

def fetch_historical_data(symbol, interval, start_date, end_date):
    """
    Fetch historical kline data from Binance in chunks to overcome limits.
    """
    all_klines = []
    while True:
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_date,
            end_date
        )
        if not klines:
            break

        all_klines.extend(klines)

        # Move start_date to last kline's close time + 1 ms
        last_close_time = klines[-1][6]
        start_date = pd.to_datetime(last_close_time, unit="ms").strftime('%Y-%m-%d %H:%M:%S')

        # Respect Binance API rate limits
        time.sleep(0.5)

        # Stop if we've reached the end date
        if pd.to_datetime(last_close_time, unit="ms") >= pd.to_datetime(end_date):
            break

    return all_klines

# Parameters
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE
start_date = "2022-01-01"  # Start from a much earlier date
end_date = "2024-12-31"

# Fetch data
klines = fetch_historical_data(symbol, interval, start_date, end_date)

# Convert to DataFrame
columns = [
    "time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]
df = pd.DataFrame(klines, columns=columns)

# Process the DataFrame
df["time"] = pd.to_datetime(df["time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
numeric_columns = ["open", "high", "low", "close", "volume",
                   "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
df[numeric_columns] = df[numeric_columns].astype(float)

# Save to CSV
df.to_csv(f"btcusd_{start_date}_to_{end_date}.csv", index=False)
print("Data fetched and saved.")
