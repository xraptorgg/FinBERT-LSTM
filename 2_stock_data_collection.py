import pandas as pd

def download_stock_data(ticker, start, end):
    """
    download stock price data from Yahoo Finance
    """
    import yfinance as yf
    stock_data = yf.download(ticker, start, end)
    df = pd.DataFrame(stock_data)
    df.to_csv("stock_price.csv")


download_stock_data("NDX", "2020-10-01", "2022-09-30")