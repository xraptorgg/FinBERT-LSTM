import pandas as pd 
import numpy as np

news_df = pd.read_csv("research/stock/news.csv")
stock_df = pd.read_csv("research/stock/stock_price.csv")

for i in range(len(stock_df)):
    date = stock_df['Date'][i][:10]
    stock_df['Date'][i] = date

news_df = news_df[news_df['Date'].isin(stock_df['Date'].tolist())]

news_df.to_csv("news_data.csv", index=False)