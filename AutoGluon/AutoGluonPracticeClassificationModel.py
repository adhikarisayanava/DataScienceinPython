import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

#Load the dataset
diabetes_df = pd.read_csv('diabetes.csv')

#EDA
'''
#check if any null values
print(diabetes_df.isnull().sum())
sns.heatmap(diabetes_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()
print(diabetes_df.info())
print(diabetes_df.describe())
print(diabetes_df.head(4))

#histogram
diabetes_df.hist(bins=30, figsize=(20,20), color='r')
plt.show()

#Correlation
diabetes_corr = diabetes_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(diabetes_corr, annot=True)
plt.show()
'''
#Split the data
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(diabetes_df, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)

#Train multiple models
predictor = TabularPredictor(label="Outcome", problem_type = 'binary', eval_metric = 'accuracy').fit(train_data = X_train, time_limit = 120, presets = "best_quality")
predictor.fit_summary()

#Evaluate tained model performance
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize = (15, 6))
sns.barplot(x = "model", y = "score_val", data = predictor.leaderboard(), color = "b")
ax.set(ylabel = "Performance Metric (Accuracy)", xlabel = "Classification Models")
plt.xticks(rotation = 45)
plt.show()

y_pred = predictor.predict(X_test) #prediction 
y_test = X_test['Outcome'] #GroundTruth

from sklearn.metrics import confusion_matrix, classification_report
plt.figure(figsize=(20,20))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_pred))