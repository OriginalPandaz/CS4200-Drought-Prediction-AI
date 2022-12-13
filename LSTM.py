import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv("spi_data.csv",index_col='month',parse_dates=True)

df.plot(figsize=(12,6))
plt.show()

# results = seasonal_decompose(df['spi1'])
# results.plot()
# plt.show()

# print(len(df))
train = df.iloc[:492]
test = df.iloc[492:]

scaler = StandardScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
print(scaled_train)
scaled_test = scaler.transform(test)

n_input = 13
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

model = Sequential()
model.add(LSTM(100,activation='relu',input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(generator,epochs=50)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
plt.show()

last_train_batch = scaled_train[-13:]
last_train_batch = last_train_batch.reshape((1,n_input,n_features))

# print(model.predict(last_train_batch))
# print(scaled_test[0])
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1,n_input,n_features ))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = np.concatenate(test_predictions).astype(None)
test['Predictions'] = true_predictions
test.plot(figsize=(12,6))
plt.show()
rmse=sqrt(mean_squared_error(test['spi1'],test['Predictions']))
print(rmse)
