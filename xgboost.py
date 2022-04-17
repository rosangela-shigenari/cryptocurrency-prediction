import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as mp
from matplotlib import pyplot
import pandas as pd
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import seaborn as sns
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

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

btcClose = result[['date','ethClose']]
btcClose = btcClose[btcClose['date'] > '2017-01-01']
btcClose_price = btcClose.copy()
print("Total data for prediction: ",btcClose_price.shape[0])

del btcClose['date']
scaler=MinMaxScaler(feature_range=(0,1))
btcClose=scaler.fit_transform(np.array(btcClose).reshape(-1,1))



training_size=int(len(btcClose)*0.75)
test_size=len(btcClose)-training_size
train_data,test_data=btcClose[0:training_size,:],btcClose[training_size:len(btcClose),:1]


time_step = 30
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

model_result = XGBRegressor(n_estimators=1000)
model_result.fit(X_train, y_train, verbose=False)

predictions = model_result.predict(X_test)
print("MAE : " + str(mean_absolute_error(y_test, predictions)))
print("RMSE : " + str(math.sqrt(mean_squared_error(y_test, predictions))))

train_predict=model_result.predict(X_train)
test_predict=model_result.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

print("Train data prediction:", train_predict.shape)
print("Test data prediction:", test_predict.shape)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))


look_back=time_step
trainPredictPlot = np.empty_like(btcClose)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(btcClose)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(btcClose)-1, :] = test_predict

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])
btcClose_price.append({'date':datetime(2022, 1, 30)}, ignore_index=True)
plotdf = pd.DataFrame({'date': btcClose_price['date'],
                       'original_close': btcClose_price['ethClose'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})


plt.plot(plotdf['date'], plotdf['original_close'], color='black', label="Current ETH close")
plt.plot(plotdf['date'], plotdf['train_predicted_close'], color='red', label="Training predicted ETH close")
plt.plot(plotdf['date'], plotdf['test_predicted_close'], color='yellow', label="Test predicted ETH close")
plt.legend()


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 1095
while(i<pred_days):
    if(len(temp_input)>time_step):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)

        yhat = model_result.predict(x_input)
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]

        lst_output.extend(yhat.tolist())
        i=i+1

    else:
        yhat = model_result.predict(x_input)

        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())

        i=i+1

print("Output of predicted next days: ", len(lst_output))

last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)

temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_year_value = temp_mat
next_predicted_year_value = temp_mat

last_original_year_value[0:time_step+1] = scaler.inverse_transform(btcClose[len(btcClose)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_year_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_year_value':last_original_year_value,
    'next_predicted_year_value':next_predicted_year_value
})

names = cycle(['Last year close price','Predicted next 3 years close price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_year_value'],
                                                      new_pred_plot['next_predicted_year_value']],
              labels={'value': 'Close price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last year vs next 3 years',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
new_pred_plot.dropna(subset = ['next_predicted_year_value','last_original_year_value'], inplace=True)
new_pred_plot.index = pd.to_datetime(new_pred_plot.index)

plt.plot(new_pred_plot.index, new_pred_plot['next_predicted_year_value'], color='yellow', label="BTC plot")
