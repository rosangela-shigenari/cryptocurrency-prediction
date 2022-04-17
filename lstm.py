import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as mp
from matplotlib import pyplot
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
py.init_notebook_mode(connected=True)
%matplotlib inline

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

def build_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - 1):
        a = dataset[i:(i + 1), 0]
        dataX.append(a)
        dataY.append(dataset[i + 1, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

def scale_feature (val):
  scaled = scaler.fit_transform(val)

  train_size = int(len(scaled) * 0.75)
  test_size = len(scaled) - train_size
  return scaled[0:train_size,:], scaled[train_size:len(scaled),:]

def build_model(train_X, test_X):
  train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
  test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

  model = Sequential()
  model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam')
  return model

scaler = MinMaxScaler(feature_range=(0, 1))

###### BTC Close #########
val = result['btcClose'].values.reshape(-1,1)
val = val.astype('float32')

train, test = scale_feature(val)

train_X, train_Y = build_dataset(train)
test_X, test_Y = build_dataset(test)

model_result = build_model(train_X, test_X)
loss = model_result.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False)
pyplot.plot(loss.history['loss'], label='train')
pyplot.plot(loss.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
yhat = model_result.predict(test_X)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
test_Y_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

predictDates = result.tail(len(test_X)).index
test_Y_reshape = test_Y_inverse.reshape(len(test_Y_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))

pyplot.plot(predictDates, test_Y_reshape, label='predict')
pyplot.plot(predictDates,yhat_reshape, label='current')
pyplot.legend()
pyplot.title('LSTM - Bitcoin Close')
pyplot.show()

rmse = math.sqrt(mean_squared_error(test_Y_inverse, yhat_inverse))
print('BTC test RMSE: %.3f' % rmse)

###### BTC Close #########
val = result['btcVolume'].values.reshape(-1,1)
val = val.astype('float32')

train, test = scale_feature(val)

train_X, train_Y = build_dataset(train)
test_X, test_Y = build_dataset(test)

model_result = build_model(train_X, test_X)

loss = model_result.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False)
pyplot.plot(loss.history['loss'], label='train')
pyplot.plot(loss.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


yhat = model_result.predict(test_X)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
test_Y_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

predictDates = result.tail(len(test_X)).index
test_Y_reshape = test_Y_inverse.reshape(len(test_Y_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))

pyplot.plot(predictDates, test_Y_reshape, label='predict')
pyplot.plot(predictDates,yhat_reshape, label='current')
pyplot.legend()
pyplot.title('LSTM - Bitcoin Volume')
pyplot.show()

rmse = math.sqrt(mean_squared_error(test_Y_inverse, yhat_inverse))
print('BTC test RMSE: %.3f' % rmse)


###### DOGE Close #########
val = result['dogeClose'].values.reshape(-1,1)
val = val.astype('float32')

train, test = scale_feature(val)

train_X, train_Y = build_dataset(train)
test_X, test_Y = build_dataset(test)

model_result = build_model(train_X, test_X)
loss = model_result.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False)
pyplot.plot(loss.history['loss'], label='train')
pyplot.plot(loss.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model_result.predict(test_X)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
test_Y_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

predictDates = result.tail(len(test_X)).index
test_Y_reshape = test_Y_inverse.reshape(len(test_Y_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))

pyplot.plot(predictDates, test_Y_reshape, label='predict')
pyplot.plot(predictDates,yhat_reshape, label='current')
pyplot.legend()
pyplot.title('LSTM - Dogecoin Close')
pyplot.show()

rmse = math.sqrt(mean_squared_error(test_Y_inverse, yhat_inverse))
print('DOGE test RMSE: %.3f' % rmse)


###### DOGE volume #########
val = result['dogeVolume'].values.reshape(-1,1)
val = val.astype('float32')

train, test = scale_feature(val)

train_X, train_Y = build_dataset(train)
test_X, test_Y = build_dataset(test)

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

model_result = build_model(train_X, test_X)
loss = model_result.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False)
pyplot.plot(loss.history['loss'], label='train')
pyplot.plot(loss.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model_result.predict(test_X)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
test_Y_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

predictDates = result.tail(len(test_X)).index
test_Y_reshape = test_Y_inverse.reshape(len(test_Y_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))

pyplot.plot(predictDates, test_Y_reshape, label='predict')
pyplot.plot(predictDates,yhat_reshape, label='current')
pyplot.legend()
pyplot.title('LSTM - Dogecoin Volume')
pyplot.show()

rmse = math.sqrt(mean_squared_error(test_Y_inverse, yhat_inverse))
print('DOGE test RMSE: %.3f' % rmse)

###### ETH Close #########
val = result['ethClose'].values.reshape(-1,1)
val = val.astype('float32')

train, test = scale_feature(val)

train_X, train_Y = build_dataset(train)
test_X, test_Y = build_dataset(test)

model_result = build_model(train_X, test_X)
loss = model_result.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False)
pyplot.plot(loss.history['loss'], label='train')
pyplot.plot(loss.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model_result.predict(test_X)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
test_Y_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

predictDates = result.tail(len(test_X)).index
test_Y_reshape = test_Y_inverse.reshape(len(test_Y_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))

pyplot.plot(predictDates, test_Y_reshape, label='predict')
pyplot.plot(predictDates,yhat_reshape, label='current')
pyplot.legend()
pyplot.title('LSTM - Ethereum Close')
pyplot.show()

rmse = math.sqrt(mean_squared_error(test_Y_inverse, yhat_inverse))
print('ETH test RMSE: %.3f' % rmse)

###### ETH volume #########
val = result['ethVolume'].values.reshape(-1,1)
val = val.astype('float32')

train, test = scale_feature(val)

train_X, train_Y = build_dataset(train)
test_X, test_Y = build_dataset(test)

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

model_result = build_model(train_X, test_X)
loss = model_result.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False)
pyplot.plot(loss.history['loss'], label='train')
pyplot.plot(loss.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
yhat = model_result.predict(test_X)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
test_Y_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

predictDates = result.tail(len(test_X)).index
test_Y_reshape = test_Y_inverse.reshape(len(test_Y_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))

pyplot.plot(predictDates, test_Y_reshape, label='predict')
pyplot.plot(predictDates,yhat_reshape, label='current')
pyplot.legend()
pyplot.title('LSTM - Ethereum Volume')
pyplot.show()

rmse = math.sqrt(mean_squared_error(test_Y_inverse, yhat_inverse))
print('ETH test RMSE: %.3f' % rmse)

sns.heatmap(result.corr(), cmap="YlGnBu", annot=True)
