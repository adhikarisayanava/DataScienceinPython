import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

######################## Read the tranining data ########################
university_df = pd.read_csv('university_admission.csv')

#first 6 rows of data
print(university_df.head(6))

#last 6 rows of data
print(university_df.tail(6))

#display the feature columns
print(university_df.columns)

#shape of dataframe
print(university_df.shape)

#check if any missing values are present in the dataframe
print(university_df.isnull().sum())

#data types of columns
print(university_df.dtypes)

#from practice
print("Highest TOEFL Score:", university_df['TOEFL_Score'].max())
print("Lowest TOEFL Score:", university_df['TOEFL_Score'].min())
print("Average TOEFL Score:", university_df['TOEFL_Score'].mean())

#if any missing values, you can drop it using:
#university_df = university_df.dropna()

######################## Perform EDA and Visualization ########################

#check if there are any null values
sns.heatmap(university_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()

#show histogram
university_df.hist(bins=30, figsize=(20,20), color = 'b')
plt.show()

#show pairplot
sns.pairplot(university_df)
plt.show()

for i in university_df.columns:
  plt.figure(figsize = (13, 7))
  sns.scatterplot(x = i, y = 'Chance_of_Admission', hue = "University_Rating", hue_norm = (1,5), data = university_df)
  plt.show()

#correlation
correlation_result = university_df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation_result,annot=True)
plt.show()  

######################## Prepare the data before model training ########################
########################Divide the data first into inputs and outputs

#Inputs
X = university_df.drop(columns='Chance_of_Admission')

#output
y = university_df['Chance_of_Admission']

#Check the shape
print(X.shape) #(1000, 7)
print(y.shape) #(1000,)

#Convert to numby array
X = np.array(X)
y = np.array(y)

# reshaping the array from (1000,) to (1000, 1)
y = y.reshape(-1,1)
y.shape
print(y.shape) #(1000, 1)

##Divide the data first into inputs and outputs
######################## Split the data into 25% for Testing and 75% for Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

print(X_train.shape) #(750, 7)
print(X_test.shape)  #(250, 7)

######################## Train and Evaluate XG-Boost model ########################
import xgboost as xgb
model = xgb.XGBRegressor(objective = 'reg:squarederror', learning_rate = 0.2, max_depth = 60, n_estimators = 400)
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

#Evaluate the model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2) 

