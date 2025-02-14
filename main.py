import time
import secerts
from binance.client import Client
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import datetime as dt
import math

# Load the model
model = load_model('my_4thmodel.keras')  # Load HDF5 format
# Test with some sample data
#for i in range(1, 101):  # Loop 100 times
 #   sample_input = np.random.rand(1, 1440, 5)  # Replace with actual test data
  #  prediction = model.predict(sample_input)
   # print("Prediction:", np.argmax(prediction, axis=1))
    #print(f"Iteration {i}")

   # time.sleep(3)  # Wait for 5 seconds
#sample_input = np.random.rand(1, 1440, 5)  # Replace with actual test data
#prediction = model.predict(sample_input)
#print("Prediction:", np.argmax(prediction, axis=1))

# API key and secret
api_key = secerts.bin_api_key
api_secret = secerts.bin_api_secret
testnet_api_key = secerts.bin_testnet_api_key
tesnet_api_secert = secerts.bin_tesnet_api_secert

#real test
#client = Client(api_key, api_secret, tld='us')
#testnet
client = Client(testnet_api_key, tesnet_api_secert, testnet=True)

start_days = ["2024-10-01","2024-09-01","2024-08-01","2024-07-01","2024-06-01","2024-05-01","2024-04-01","2024-03-01","2024-02-01","2024-01-01"]
end_days = ["2024-10-31","2024-09-30","2024-08-31","2024-07-31","2024-06-30","2024-05-31","2024-04-30","2024-03-31","2024-02-28","2024-01-31"]


# Fetch account info

symbol = "BTCUSDT"
buy_price_thershold = 60000
sell_price_thershold = 68000
trade_quantity = 0.001

# Get trading rules for BTCUSDT
symbol_info = client.get_symbol_info(symbol)

# Extract step size (minimum quantity increment)
step_size = float([f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0]['stepSize'])

# Function to round quantity down to valid step size
def round_step_size(value, step_size):
    return math.floor(value / step_size) * step_size

def get_current_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])


def place_buy_order(symbol,quantity):
    curprice = get_current_price(symbol)
    quantity = quantity / curprice
    quantity = round_step_size(quantity, step_size)
    order = client.order_market_buy(symbol=symbol, quantity = quantity)
    print(f"bought: {order}")
#place_buy_order(symbol,trade_quantity)

def place_sell_order(symbol,quantity):
    curprice = get_current_price(symbol)
    quantity = quantity / curprice
    quantity = round_step_size(quantity, step_size)
    order = client.order_market_sell(symbol=symbol,quantity = quantity)
    print(f"sold: {order}")

def account_info():
    account_info = client.get_account()
    print(f"account info: {account_info}")
#place_sell_order(symbol,trade_quantity)

def tradingbot():
    inpos = False;

    while True:
        curprice = get_current_price(symbol)
        print(f"current price of {symbol}: {curprice}")
        # Fetch the last 24 hours of minute-by-minute data
        #symbol = 'BTCUSDT'
        interval = Client.KLINE_INTERVAL_1MINUTE
        limit = 1440  # Request 1440 minutes (24 hours)

        # Binance API allows `limit` up to 1000 per call, so fetch in batches if needed
        if limit > 1000:
            klines = []
            for start in range(0, limit, 1000):
                batch = client.get_klines(symbol=symbol, interval=interval, limit=min(1000, limit - start))
                klines.extend(batch)
        else:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

        # Convert data into a DataFrame for easier processing
        columns = ["time", "open", "high", "low", "close", "volume", "close_time",
                   "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
        df = pd.DataFrame(klines, columns=columns)

        # Convert relevant columns to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        # Process into the required categories
        df_processed = pd.DataFrame()
        df_processed["close"] = df["close"]
        df_processed["range"] = df["high"] - df["low"]  # High-Low range
        df_processed["change"] = df["close"] - df["open"]  # Close-Open change
        df_processed["volatility"] = (df["high"] - df["low"]) / df["open"]  # Relative volatility
        df_processed["volume"] = df["volume"]

        # Reset the index for clean DataFrame output
        df_processed.reset_index(drop=True, inplace=True)

        # Display the first few rows
        print(df_processed.head())

        # Convert to NumPy array if required
        data_array = df_processed.to_numpy()
        print(data_array.shape)  # Shape: (1440, 5)
        data_array = data_array.reshape(1, 1440, 5)
        prediction = model.predict(data_array)  # Returns an array
        predicted_int = int(np.argmax(prediction, axis=1)[0])  # Convert first element to int

        print("Prediction:", predicted_int)
        if(predicted_int == 0):
            place_sell_order(symbol,20)
        elif(predicted_int == 1):
            place_buy_order(symbol,20)


        #print("Prediction:", prediction)
        timme = dt.datetime.now()
        usdt_balance = float(client.get_asset_balance(asset="USDT")['free'])
        btc_balance = float(client.get_asset_balance(asset="BTC")['free'])
        totval = usdt_balance + (btc_balance * curprice)
        # Data to append (as a DataFrame)
        #print(usdt_balance['free'])
        #print(btc_balance['free'])
        dfff = pd.DataFrame([[timme, curprice, usdt_balance,btc_balance,totval,predicted_int]], columns=["datetime","BitPrice", "USDBal", "BitBal","totalval","Prediction"])

        # Append to the CSV file without overwriting (mode='a' and header=False to avoid duplicate headers)
        dfff.to_csv("testingoutput.csv", mode="a", index=False, header=False)






        time.sleep(900)

tradingbot()




