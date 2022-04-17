import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt

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

np.random.seed(6789)
fig = plt.figure(figsize = (8,4))
ax = fig.gca()
x = bitcoinDf["BTC close"]
result = plt.hist(x, bins=20, color='c', edgecolor='k', alpha=0.65)
plt.title('BTC close histogram')
plt.axvline(x.mean(), color='red', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(x.mean()*1.1, max_ylim*0.9, 'Avg: {:.2f}'.format(x.mean()))


fig = plt.figure(figsize = (8,4))
ax = fig.gca()
x = bitcoinDf["BTC volume"]
result = plt.hist(x, bins=20, color='c', edgecolor='k', alpha=0.65)
plt.title('BTC volume histogram')
plt.axvline(x.mean(), color='red', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(x.mean()*1.1, max_ylim*0.9, 'Avg: {:.2f}'.format(x.mean()))


fig = plt.figure(figsize = (8,4))
ax = fig.gca()
x = dogecoinDf["DOGE close"]
result = plt.hist(x, bins=20, color='c', edgecolor='k', alpha=0.65)
plt.title('DOGE close histogram')
plt.axvline(x.mean(), color='red', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(x.mean()*1.1, max_ylim*0.9, 'Avg: {:.2f}'.format(x.mean()))


fig = plt.figure(figsize = (8,4))
ax = fig.gca()
x = dogecoinDf["DOGE volume"]
result = plt.hist(x, bins=20, color='c', edgecolor='k', alpha=0.65)
plt.title('DOGE volume histogram')
plt.axvline(x.mean(), color='red', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(x.mean()*1.1, max_ylim*0.9, 'Avg: {:.2f}'.format(x.mean()))

fig = plt.figure(figsize = (8,4))
ax = fig.gca()
x = ethereumDf["ETH close"]
result = plt.hist(x, bins=20, color='c', edgecolor='k', alpha=0.65)
plt.title('ETH close histogram')
plt.axvline(x.mean(), color='red', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(x.mean()*1.1, max_ylim*0.9, 'Avg: {:.2f}'.format(x.mean()))


fig = plt.figure(figsize = (8,4))
ax = fig.gca()
x = ethereumDf["ETH volume"]
result = plt.hist(x, bins=20, color='c', edgecolor='k', alpha=0.65)
plt.title('ETH volume histogram')
plt.axvline(x.mean(), color='red', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(x.mean()*1.1, max_ylim*0.9, 'Avg: {:.2f}'.format(x.mean()))
