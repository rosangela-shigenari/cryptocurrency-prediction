import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

pd.set_option('display.expand_frame_repr', False)
#### Bitcoin dataset data cleaning ####
bitcoinDf = pd.read_csv('coin_Bitcoin.csv')

# Exclude unused columns
bitcoinDf.drop('High', inplace=True, axis=1)
bitcoinDf.drop('Low', inplace=True, axis=1)
bitcoinDf.drop('Open', inplace=True, axis=1)
bitcoinDf.drop('Marketcap', inplace=True, axis=1)
bitcoinDf.drop('SNo', inplace=True, axis=1)
bitcoinDf.drop('Name', inplace=True, axis=1)
bitcoinDf.drop('Symbol', inplace=True, axis=1)

#Select 5 last years
bitcoinDf = bitcoinDf[bitcoinDf['Date'].str.contains('2017|2018|2019|2020|2021')]
bitcoinDf['Date'] = pd.to_datetime(bitcoinDf['Date'].str[:10], format='%Y-%m-%d')
bitcoinDf = bitcoinDf.rename(columns = {'Close': 'BTC close', 'Volume': 'BTC volume'}, inplace = False)
print('###### Bitcoin data after clean up #####')
print(bitcoinDf)
print('###### #####')

#### DogeCoin dataset data cleaning ####
dogecoinDf = pd.read_csv('coin_Dogecoin.csv')
# Exclude unused columns
dogecoinDf.drop('High', inplace=True, axis=1)
dogecoinDf.drop('Low', inplace=True, axis=1)
dogecoinDf.drop('Open', inplace=True, axis=1)
dogecoinDf.drop('Marketcap', inplace=True, axis=1)
dogecoinDf.drop('SNo', inplace=True, axis=1)
dogecoinDf.drop('Name', inplace=True, axis=1)
dogecoinDf.drop('Symbol', inplace=True, axis=1)

#Select 5 last years
dogecoinDf = dogecoinDf[dogecoinDf['Date'].str.contains('2017|2018|2019|2020|2021')]
dogecoinDf['Date'] = pd.to_datetime(dogecoinDf['Date'].str[:10], format='%Y-%m-%d')
dogecoinDf = dogecoinDf.rename(columns = {'Close': 'DOGE close', 'Volume': 'DOGE volume'}, inplace = False)

print('###### Dogecoin data after clean up #####')
print(dogecoinDf)
print('###### #####')



#### Ethereum dataset data cleaning ####
ethereumDf = pd.read_csv('coin_Ethereum.csv')
# Exclude unused columns
ethereumDf.drop('High', inplace=True, axis=1)
ethereumDf.drop('Low', inplace=True, axis=1)
ethereumDf.drop('Open', inplace=True, axis=1)
ethereumDf.drop('Marketcap', inplace=True, axis=1)
ethereumDf.drop('SNo', inplace=True, axis=1)
ethereumDf.drop('Name', inplace=True, axis=1)
ethereumDf.drop('Symbol', inplace=True, axis=1)

#Select 5 last years
ethereumDf = ethereumDf[ethereumDf['Date'].str.contains('2017|2018|2019|2020|2021')]
ethereumDf['Date'] = pd.to_datetime(ethereumDf['Date'].str[:10], format='%Y-%m-%d')
ethereumDf = ethereumDf.rename(columns = {'Close': 'ETH close', 'Volume': 'ETH volume'}, inplace = False)

print('###### Ethereum data after clean up #####')
print(ethereumDf)
print('###### #####\n\n')


#### S&P 500 dataset data cleaning ####
sp500df = yf.download('^GSPC',
                      start='2017-01-11',
                      end='2021-12-31',
                      progress=True,
)
sp500df.head()

# Exclude unused columns
sp500df.drop('High', inplace=True, axis=1)
sp500df.drop('Low', inplace=True, axis=1)
sp500df.drop('Open', inplace=True, axis=1)
sp500df.drop('Adj Close', inplace=True, axis=1)
sp500df = sp500df.rename(columns = {'Close': 'S&P 500 close', 'Volume': 'S&P 500 volume'}, inplace = False)
sp500df.reset_index(inplace=True)
print(sp500df)
#### Join datasets procedure ####

result = pd.merge(bitcoinDf, dogecoinDf, how="outer", on=["Date"])
result = pd.merge(result, ethereumDf, how="outer", on=["Date"])
result = pd.merge(result, sp500df, how="outer", on=["Date"])


result = result[['Date','BTC close', 'DOGE close', 'ETH close', 'S&P 500 close', 'BTC volume', 'DOGE volume', 'ETH volume','S&P 500 volume']]
result.dropna(subset = ['Date','BTC close', 'DOGE close', 'ETH close', 'S&P 500 close', 'BTC volume', 'DOGE volume', 'ETH volume','S&P 500 volume'], inplace=True)
result.head(20)

result.describe()

#### Export to csv file ####
result.to_csv(r'dataset.csv', index=False)
