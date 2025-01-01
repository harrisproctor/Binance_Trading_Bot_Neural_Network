import time
import secerts
import pandas as pd
from binance.client import Client

# Replace with your API key and secret
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



def get_current_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])


def place_buy_order(symbol,quantity):
    order = client.order_market_buy(symbol=symbol, quantity = quantity)
    print(f"bought: {order}")
#place_buy_order(symbol,trade_quantity)

def place_sell_order(symbol,quantity):
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




        time.sleep(3)

#tradingbot()


