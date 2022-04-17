import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as mp
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import seaborn as sb
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
bitcoinDf['date'] = pd.to_datetime(bitcoinDf['Date'].str[:10], format='%Y-%m-%d')
bitcoinDf = bitcoinDf.rename(columns = {'Close': 'btcClose', 'Volume': 'btcVolume'}, inplace = False)

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
dogecoinDf = dogecoinDf[dogecoinDf['Date'].str.contains('2021')]
dogecoinDf['date'] = pd.to_datetime(dogecoinDf['Date'].str[:10], format='%Y-%m-%d')
dogecoinDf = dogecoinDf.rename(columns = {'Close': 'dogeClose', 'Volume': 'dogeVolume'}, inplace = False)


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
ethereumDf = ethereumDf[ethereumDf['Date'].str.contains('2021')]
ethereumDf['date'] = pd.to_datetime(ethereumDf['Date'].str[:10], format='%Y-%m-%d')
ethereumDf = ethereumDf.rename(columns = {'Close': 'ethClose', 'Volume': 'ethVolume'}, inplace = False)


#### S&P 500 dataset data cleaning ####
sp500df = yf.download('^GSPC',
                      start='2017-01-01',
                      end='2021-05-01',
                      progress=True,
)
sp500df.head()

# Exclude unused columns
sp500df.drop('High', inplace=True, axis=1)
sp500df.drop('Low', inplace=True, axis=1)
sp500df.drop('Open', inplace=True, axis=1)
sp500df.drop('Adj Close', inplace=True, axis=1)
sp500df = sp500df.rename(columns = {'Close': 'spClose', 'Volume': 'spVolume'}, inplace = False)
sp500df.reset_index(inplace=True)
sp500df['date'] = sp500df['Date']

result = pd.merge(bitcoinDf, dogecoinDf, how="outer", on=["date"])
result = pd.merge(result, ethereumDf, how="outer", on=["date"])
result = pd.merge(result, sp500df, how="outer", on=["date"])


result = result[['date','btcClose', 'dogeClose', 'ethClose', 'spClose', 'btcVolume', 'dogeVolume', 'ethVolume','spVolume']]
result.dropna(subset = ['date','btcClose', 'dogeClose', 'ethClose', 'spClose', 'btcVolume', 'dogeVolume', 'ethVolume','spVolume'], inplace=True)

x_data = result.drop(['date'], axis=1)

X = result[['btcClose', 'dogeClose', 'ethClose', 'spClose', 'btcVolume', 'dogeVolume', 'ethVolume','spVolume']].values
y = result['ethClose'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1)

regressor = LinearRegression()

regressor.fit(X_train, y_train)
print(regressor.coef_)
predicted = regressor.predict(X_test)

data1 = pd.DataFrame({'current': y_test.flatten(), 'predicted' : predicted.flatten()})

print('(*) ETH Close - mean absolute error:', metrics.mean_absolute_error(y_test,predicted))
print('(*) ETH Close - mean squared error:', metrics.mean_squared_error(y_test,predicted))
print('(*) ETH Close - root mean squared error:', math.sqrt(metrics.mean_squared_error(y_test,predicted)))
