import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


bike_df = pd.read_csv('bike_sharing_daily.csv')

###########################EDA
'''
print(bike_df.info())
print(bike_df.head(3))
print(bike_df.describe())


#check if null values exists
print(bike_df.isnull().sum())
sns.heatmap(bike_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.show()

'''
#Removing columns which are not needed
bike_df = bike_df.drop(labels = ['instant','casual', 'registered'], axis = 1)
#print(bike_df.info())

#Below lines will help eventually in data visualization
bike_df['dteday'] = pd.to_datetime(bike_df['dteday'], format = '%m/%d/%Y')
bike_df.index = pd.DatetimeIndex(bike_df['dteday']) #will change the index to dteday
bike_df = bike_df.drop(labels = ['dteday'], axis = 1)
#print(bike_df.head(6))

###########################Data Visualization
#by week
plt.figure(figsize = (12, 7))
bike_df['cnt'].asfreq('W').plot(linewidth = 5)
plt.title('Bike Rental Usage Per Week')
plt.xlabel('Week')
plt.ylabel('Bike Rental')
plt.grid()
plt.show()

#by month
plt.figure(figsize = (12, 7))
bike_df['cnt'].asfreq('M').plot(linewidth = 5)
plt.title('Bike Rental Usage Per Month')
plt.xlabel('Week')
plt.ylabel('Bike Rental')
plt.grid()
plt.show()

#by quarter
#by month
plt.figure(figsize = (12, 7))
bike_df['cnt'].asfreq('Q').plot(linewidth = 6)
plt.title('Bike Rental Usage Per Quarter')
plt.xlabel('Week')
plt.ylabel('Bike Rental')
plt.grid()
plt.show()

#Correlation heatmap
X_numerical = bike_df[['temp', 'hum', 'windspeed', 'cnt']]
plt.figure(figsize=(20,12))
sns.heatmap(X_numerical.corr(), annot=True)
plt.show()

###########################Create training and testing dataset
X_cat = bike_df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]
#print(X_cat)
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray() #converting it to numpy array
print(X_cat.shape)

X_cat = pd.DataFrame(X_cat) #converting back again to data frame
#print(X_cat.head(4))
#print(X_numerical)
X_numerical = X_numerical.reset_index() #converting back to normal index from dteday index
print(X_numerical)

X_all = pd.concat([X_cat, X_numerical], axis = 1)
X_all = X_all.drop('dteday', axis = 1) #drop it now since we have month, yr columns already

#split the data using the new way below
#X = X_all.iloc[:, :-1].values
#y = X_all.iloc[:, -1:].values
#OR
X = X_all.drop(columns=['cnt'])
y = X_all[['cnt']]

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
 
###########################Train and evaluate the model using XG-Boost 
import xgboost as xgb
from math import sqrt

#Train the model
model = xgb.XGBRegressor(objective = 'reg:squarederror', learning_rate = 0.2, max_depth = 8, n_estimators = 500)

model.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
result = model.score(X_test, y_test)
print(f"Accuracy: {result}")

y_predict = model.predict(X_test)

#Evaluate the model
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2) 
