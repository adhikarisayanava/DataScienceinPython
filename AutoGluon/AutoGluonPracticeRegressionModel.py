import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

insurance_df = pd.read_csv('insurance.csv')

######################## EDA ########################

#print(insurance_df.info())
#print(insurance_df.head(4))
#print(insurance_df.isnull().sum())
print("Unique regions in the data frame are:", insurance_df['region'].unique())
#print(insurance_df.describe())
'''
# Grouping by region to see any relationship between region and charges
# Seems like south east region has the highest charges and body mass index
df_region = insurance_df.groupby(by='region').mean(numeric_only=True)
#print(df_region)

age = insurance_df.groupby(by='age').mean(numeric_only=True)
#print(age)

#Check if there are any null values
print(insurance_df.isnull().sum())
#OR
sns.heatmap(insurance_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.show()

#histogram
insurance_df[['age', 'sex', 'bmi', 'children', 'smoker', 'charges']].hist(bins = 30, figsize = (12, 12), color = 'r')
plt.show()

#pairplot
sns.pairplot(insurance_df)
plt.show()

#regression plot
plt.figure(figsize = (15, 6))
sns.regplot(x = 'age', y = 'charges', data = insurance_df)
plt.show()

# Correlation Matrix
corr_matrix = insurance_df.corr(numeric_only=True)
plt.figure(figsize = (15, 15))
cm = sns.heatmap(corr_matrix,
               linewidths = 1,
               annot = True, 
               fmt = ".2f")
plt.title("Correlation Matrix of Insurance charges", fontsize = 20)
plt.show()

'''
######################## Split the data ########################
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(insurance_df, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)

######################## Train the model ########################
# Train multiple ML regression models using AutoGluon
# You need to specify the target column, train_data, limit_time, and presets 
# Note that AutoGluon automatically detects if the problem is classification or regression type problems from the 'label' column
# For regression type problems, 'label' values are generally floating point non-integers with large number of unique values

predictor = TabularPredictor(label="charges", problem_type = 'regression', eval_metric = 'r2').fit(train_data = X_train, time_limit = 100, presets = "best_quality")
predictor.fit_summary()

######################## Evaluate the model ########################
predictor.leaderboard()

f, ax = plt.subplots(figsize = (15, 6))
sns.barplot(x="model", y="score_val", data=predictor.leaderboard(), color="b")
ax.set(ylabel = "Performance Metric (R2)", xlabel = "Regression Models")
plt.xticks(rotation = 45)
plt.show()

y_test = X_test['charges'] #groundtruth
y_predict = predictor.predict(X_test)
plt.figure(figsize = (15, 10))
plt.plot(y_test, y_predict, "^", color = 'r')
plt.ylabel('Model Predictions')
plt.xlabel('True Values')
plt.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2) 
