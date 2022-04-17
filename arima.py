import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as mp
from matplotlib import pyplot
import pandas as pd
import yfinance as yf
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')

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
dogecoinDf = dogecoinDf[dogecoinDf['Date'].str.contains('2017|2018|2019|2020|2021')]
dogecoinDf['date'] = pd.to_datetime(dogecoinDf['Date'].str[:10], format='%Y-%m-%d')
dogecoinDf = dogecoinDf.rename(columns = {'Close': 'dogeClose', 'Volume': 'dogeVolume'}, inplace = False)


#### Ethereum dataset data cleaning ####
ethereumDf = pd.read_csv('coin_Bitcoin.csv')
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
result.index = result['date']

result_month = result.resample('M').mean()

plt.figure(figsize=[15,7])

sm.tsa.seasonal_decompose(result_month['ethClose']).plot()
print("ARIMA diagnostics: BTC Close: p=%f" % sm.tsa.stattools.adfuller(result_month['ethClose'])[1])


result_month['ethClose_box'], lmbda = stats.boxcox(result_month['ethClose'])
sm.tsa.stattools.adfuller(result_month['ethClose'])[1]

# Seasonal differentiation
result_month['ethClose_seasonal_diff'] = result_month['ethClose_box'] - result_month['ethClose_box'].shift(12)
print("ARIMA diagnostics: BTC Close: p=%f" % sm.tsa.stattools.adfuller(result_month['ethClose_regular_diff'][12:])[1])


# Regular differentiation
result_month['ethClose_regular_diff'] = result_month['ethClose_seasonal_diff'] - result_month['ethClose_seasonal_diff'].shift(1)
plt.figure(figsize=(15,7))

# STL-decomposition
sm.tsa.seasonal_decompose(result_month['ethClose_regular_diff'][13:]).plot()
print("ARIMA diagnostics: BTC Close: p=%f" % sm.tsa.stattools.adfuller(result_month['ethClose_regular_diff'][13:])[1])

# Regular differentiation
result_month['ethClose_regular_diff'] = result_month['ethClose_seasonal_diff'] - result_month['ethClose_seasonal_diff'].shift(1)
plt.figure(figsize=(15,7))

# STL-decomposition
sm.tsa.seasonal_decompose(result_month['ethClose_regular_diff'][13:]).plot()
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(result_month['ethClose_regular_diff'][13:])[1])

# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(result_month['ethClose_regular_diff'][13:].values.squeeze(), lags=36, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(result_month['ethClose_regular_diff'][13:].values.squeeze(), lags=36, ax=ax)
plt.tight_layout()

# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(result_month['ethClose'], order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())

# STL-decomposition
plt.figure(figsize=(15,7))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=36, ax=ax)

print("Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

plt.tight_layout()
plt.title('ARIMA diagnostics: BTC Close')
plt.show()

# Inverse Box-Cox Transformation Function
def inverse_box_cox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))
# Prediction

df_month2 = result_month[['ethClose']]
date_list = [datetime(2021, 6, 30), datetime(2021, 7, 31), datetime(2021, 8, 31), datetime(2021, 9, 30),
             datetime(2021, 10, 31), datetime(2021, 11, 30), datetime(2021, 12, 31), datetime(2022, 1, 31),
             datetime(2022, 2, 28), datetime(2022, 3, 31), datetime(2022, 4, 30), datetime(2022, 5, 31),
             datetime(2022, 6, 30), datetime(2022, 7, 31), datetime(2022, 8, 31), datetime(2022, 9, 30),
             datetime(2022, 10, 31), datetime(2022, 11, 30), datetime(2022, 12, 31),datetime(2023, 1, 31),
             datetime(2023, 2, 28), datetime(2023, 3, 31), datetime(2023, 4, 30), datetime(2023, 5, 31),
             datetime(2023, 6, 30), datetime(2023, 7, 31), datetime(2023, 8, 31), datetime(2023, 9, 30),
             datetime(2023, 10, 31), datetime(2023, 11, 30), datetime(2023, 12, 31)]
future = pd.DataFrame(index=date_list, columns= result_month.columns)
df_month2 = pd.concat([df_month2, future])
df_month2['forecast'] = inverse_box_cox(best_model.predict(start=0, end=75), lmbda)
plt.figure(figsize=(15,7))
df_month2.ethClose.plot()
df_month2.forecast.plot(color='r', ls='--', label='Predicted ethClose')
plt.legend()
plt.title('Ethereum Close')
plt.ylabel('price USD')
plt.show()
