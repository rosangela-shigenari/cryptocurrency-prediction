import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

pd.set_option('display.expand_frame_repr', False)

#### S&P 500 dataset data cleaning ####
sp500df = yf.download('^GSPC',
                      start='2017-01-11',
                      end='2021-12-31',
                      progress=True,
)
# Exclude unused columns
sp500df.drop('High', inplace=True, axis=1)
sp500df.drop('Low', inplace=True, axis=1)
sp500df.drop('Open', inplace=True, axis=1)
sp500df.drop('Adj Close', inplace=True, axis=1)
sp500df = sp500df.rename(columns = {'Close': 'S&P 500 close', 'Volume': 'S&P 500 volume'}, inplace = False)
sp500df.reset_index(inplace=True)

formatter = mdates.DateFormatter("%Y")
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
locator = mdates.YearLocator()
ax.xaxis.set_major_locator(locator)
ax.set_ylabel('Volume of transactions')
ax.set_xlabel('Years')

plt.plot(data["Date"], data["Volume"], label="Ethereum plot")

plt.show()
