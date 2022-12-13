import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

df = pd.read_csv("linear_data.csv")
# print(df)

# sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
# plt.show()

n = 70
split = int(len(df)*(n/100))
training = df[:split]
test = df[split:]

plt.figure(figsize=(12,6))
plt.scatter(training['month'],training['spi1'])
plt.title('Month Vs SPI 1')
plt.xlabel('Month')
plt.ylabel('SPI 1')
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')
plt.show()


new_df = training.drop('spi1',axis='columns')
SPI1 = training['spi1']

lr = linear_model.LinearRegression()
lr.fit(new_df,SPI1)
predict_month = 516
month_prediction = lr.predict([[predict_month]])
print(month_prediction)

y_predict = lr.predict(new_df)
plt.figure(figsize=(12,6))
plt.scatter(training['month'],training['spi1'])
plt.plot(training['month'],y_predict, 'g')
plt.show()

print('Intercept: ',lr.intercept_)
print('Coefficient: ',lr.coef_)
