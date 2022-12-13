import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from pmdarima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset

df = pd.read_csv("spi_data.csv")
df['month'] = pd.to_datetime(df['month'], infer_datetime_format=True)
indexDataset = df.set_index(['month'])
print(indexDataset.shape)
# df['spi1'].plot(figsize=(12,6))
# plt.show()

# results = seasonal_decompose(indexDataset['spi1'])
# results.plot()
# plt.show()

print('Results of Dickey-Fuller Test')
dftest = adfuller(indexDataset['spi1'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistics', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print(dfoutput)

warnings.filterwarnings("ignore")
stepwise_fit = auto_arima(indexDataset['spi1'],trace=True,suppress_warnings=True, m=12)
stepwise_fit.summary()

train = indexDataset.iloc[:-24]
test = indexDataset.iloc[-24:]

model = sm.tsa.statespace.SARIMAX(train['spi1'],order=(1,1,1),seasonal_order=(2,3,2,12))
results = model.fit()
model = model.fit()

indexDataset['forecast']=results.predict(start=492,end=516,dynamic=True)
indexDataset[['spi1','forecast']].plot(figsize=(12,6))
plt.show()

pred = model.predict(start=len(train),end=len(indexDataset)-1,typ='levels')

test['spi1'].plot(legend=True)
pred.plot(legend=True)
plt.show()

print(test['spi1'].mean())
rmse=sqrt(mean_squared_error(pred,test['spi1']))
print(rmse)
