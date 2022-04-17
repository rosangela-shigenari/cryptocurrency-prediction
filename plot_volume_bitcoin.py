import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_csv('coin_Bitcoin.csv')

data = data[data["Date"].str.contains("2017|2018|2019|2020|2021")]
data['Date'] = pd.to_datetime(data['Date'].str[:10], format='%Y-%m-%d')


formatter = mdates.DateFormatter("%Y")
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
locator = mdates.YearLocator()
ax.xaxis.set_major_locator(locator)
ax.set_ylabel('Volume of transactions')
ax.set_xlabel('Years')

plt.plot(data["Date"], data["Volume"], label="Bitcoin plot")

plt.show()
