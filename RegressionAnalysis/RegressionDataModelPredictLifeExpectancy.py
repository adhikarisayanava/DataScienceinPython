import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

######################## Read the tranining data ########################
life_expectancy_df = pd.read_csv('Life_Expectancy_Data.csv')
'''
#print(life_expectancy_df.head(6))
print(life_expectancy_df.columns)
print(life_expectancy_df.isnull().sum())
print(life_expectancy_df.shape)
print(life_expectancy_df.info())
print(life_expectancy_df.describe())

#histogram
life_expectancy_df.hist(bins=30, figsize=(10,10), color = 'r')
plt.show()

#pairplot
plt.figure(figsize=(20,20))
sns.pairplot(life_expectancy_df)
plt.show()

#correlation matrix
plt.figure(figsize=(12,12))
corr_matrix = life_expectancy_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True)
plt.show()
'''
######################## Perform EDA and Visualization ########################
'''
#check if there are any null values using heatmap
sns.heatmap(life_expectancy_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()

#scatterplot
sns.scatterplot(x = 'Income composition of resources', y = 'Life expectancy ', hue = 'Status', data = life_expectancy_df)
plt.show()
'''

#checking the unique value in country to consider it as a categorical value
print(life_expectancy_df['Status'].nunique())
#Perform one hot encoding
life_expectancy_df = pd.get_dummies(life_expectancy_df, columns = ['Status'], dtype=int)
#print(life_expectancy_df.head(2))

# Check the number of null values for the columns having null values
#print(life_expectancy_df.isnull().sum()[np.where(life_expectancy_df.isnull().sum() != 0)[0]])

# Since most of the are continous values we replace them with mean
life_expectancy_df = life_expectancy_df.apply(lambda x: x.fillna(x.mean()),axis=0)
#print(life_expectancy_df.isnull().sum())

######################## Prepare the data before model training ########################
########################Divide the data first into inputs and outputs
X = life_expectancy_df.drop(columns= ['Life expectancy '])
y = life_expectancy_df[['Life expectancy ']]

print(X.shape)
print(y.shape)

#Convert the data type to float32
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

#Split the training and test data into 25% for Testing and 75% for Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape) #(2350, 7)
print(X_test.shape)  #(588, 7)

######################## Train and Evaluate XG-Boost model ########################
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from math import sqrt
import xgboost as xgb

#Train the model
model = xgb.XGBRegressor(objective = 'reg:squarederror', learning_rate = 0.2, max_depth = 2, n_estimators = 100)
#Hyperparameters:
    #learning_rate = how aggressive we want to train the model
    #max_depth = depth of tree created while training the model
    #n_estimators = number of boosting rounds
model.fit(X_train, y_train)

# predict the score of the trained model using the testing dataset
result = model.score(X_test, y_test)
print("Accuracy : {}".format(result))

#make prediction on the test data
y_predict = model.predict(X_test)

#plot the scaled result to compare
plt.plot(y_test, y_predict, "*" , color = 'r')
plt.xlabel('Model predictions')
plt.ylabel('True Values')
plt.show()

#Evaluate the model
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2) 