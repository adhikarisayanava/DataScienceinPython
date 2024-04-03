import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

######################## Read the tranining data ########################
#Read the csv file

creditcard_df = pd.read_csv('UCI_Credit_Card.csv')
print(creditcard_df.info())
print(creditcard_df.describe())
print(creditcard_df.isnull().sum())

######################## Perform EDA and Visualization ########################
'''
#histogram
creditcard_df.hist(bins = 30, figsize=(20,20), color='r')
plt.show()

#pie chart
creditcard_df["default.payment.next.month"].value_counts()
plt.figure(figsize = [10, 10])
creditcard_df["default.payment.next.month"].value_counts().plot(kind='pie')
plt.show()

# Correlation Matrix
corr_matrix = creditcard_df.corr()
plt.figure(figsize = (15, 15))
cm = sns.heatmap(corr_matrix,
               linewidths = 1,
               annot = True, 
               fmt = ".2f")
plt.title("Correlation Matrix of Credit Card Customers", fontsize = 20)
plt.show()
'''
######################## Prepare Data before model training ########################
#1)First check if there is any null values for any row/columns and think whether to remove or replace such entries
print(creditcard_df.isnull().sum())

#2)Remove probably columns which does not have any impact/value on the model and can consume unnecessary memory, cpu.
creditcard_df.drop(['ID'], axis = 1, inplace = True)

#3)Check for balance/imbalance in the data set
print("Total =", len(creditcard_df))
cc_default_df = creditcard_df[creditcard_df['default.payment.next.month'] == 1]
print("Number of customers who defaulted on their credit card payments =", len(cc_default_df))
cc_nodefault_df = creditcard_df[creditcard_df['default.payment.next.month'] == 0]
print("Number of customers who did not default on their credit card payments (paid their balance)=", len(cc_nodefault_df))

#4)Do onehotencoder data preprocessing to convert categorical data into a format that works better with machine learning algorithms(for example in binary code)
#Split input variables
X_cat = creditcard_df[['SEX', 'EDUCATION', 'MARRIAGE']]
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
print(X_cat.shape)
X_cat = pd.DataFrame(X_cat)

#5)Data preprocessing like scaling
X_numerical = creditcard_df[['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
X = pd.concat([X_cat, X_numerical], axis = 1)
#print(X.head(4))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X.columns = X.columns.astype(str)
X = scaler.fit_transform(X)
print("-------------------------------------------------------------------------------------------------\n")
#print(X)

#6)split output variable
y = creditcard_df['default.payment.next.month']

########################Perform train/test split########################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

################################################ Train the model ################################################
#XG-BOOST CLASSIFIER
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

model_XGB = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 20, use_label_encoder = False)
model_XGB.fit(X_train,y_train.values.ravel())

y_predict = model_XGB.predict(X_test)
print("Prediction from XGBoost classifier \n")
print(classification_report(y_test, y_predict))

#plot the confusion matrix
#cm = confusion_matrix(y_test, y_predict)
#sns.heatmap(cm, annot=True, fmt=".4g")
#plt.show()


#LOGISTIC REGRESSION CLASSIFIER
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression(max_iter=3000)
model_LR.fit(X_train, y_train.values.ravel())

y_predict = model_LR.predict(X_test)
print("Prediction from Logistic Regression classifier \n")
print(classification_report(y_test, y_predict))

#plot the confusion matrix
#cm = confusion_matrix(y_test, y_predict)
#sns.heatmap(cm, annot=True, fmt=".4g")
#plt.show()


#SUPPORT VECTOR MACHINE CLASSIFIER
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

#Below portion of code is to suppress Warning during training of the SVM model
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=FutureWarning)

model_svc = LinearSVC(max_iter=10000)
model_svm = CalibratedClassifierCV(model_svc)
model_svm.fit(X_train,y_train.values.ravel())

y_predict = model_svm.predict(X_test)

print("Prediction from Support Vector Machine classifier \n")
print(classification_report(y_test, y_predict))

#plot the confusion matrix
#cm = confusion_matrix(y_test, y_predict)
#sns.heatmap(cm, annot=True, fmt=".4g")
#plt.show()


#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train.values.ravel())

y_predict = model_rf.predict(X_test)
print("Prediction from Random Forest classifier \n")
print(classification_report(y_test, y_predict))

#plot the confusion matrix
#cm = confusion_matrix(y_test, y_predict)
#sns.heatmap(cm, annot=True, fmt=".4g")
#plt.show()


#K-Nearest Neighbour classifier

from sklearn.neighbors import KNeighborsClassifier

model_knn =  KNeighborsClassifier()
model_knn.fit(X_train, y_train.values.ravel())

y_predict = model_knn.predict(X_test)
print("Prediction from KNN classifier \n")
print(classification_report(y_test, y_predict))

#plot the confusion matrix
#cm = confusion_matrix(y_test, y_predict)
#sns.heatmap(cm, annot=True, fmt=".4g")
#plt.show()


#Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train.values.ravel())

y_predict = model_gnb.predict(X_test)
print("Prediction from Naive Bayes classifier \n")
print(classification_report(y_test, y_predict))

#plot the confusion matrix
#cm = confusion_matrix(y_test, y_predict)
#sns.heatmap(cm, annot=True, fmt=".4g")
#plt.show()


################################################################################################
#Plot ROC curve and evaluate AUC for all the 5 classifier models from above
from sklearn.metrics import roc_curve
fpr1, tpr1, thresh1 = roc_curve(y_test, model_LR.predict_proba(X_test)[:, 1], pos_label = 1)
fpr2, tpr2, thresh2 = roc_curve(y_test, model_svm.predict_proba(X_test)[:, 1], pos_label = 1)
fpr3, tpr3, thresh3 = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1], pos_label = 1)
fpr4, tpr4, thresh4 = roc_curve(y_test, model_knn.predict_proba(X_test)[:, 1], pos_label = 1)
fpr5, tpr5, thresh5 = roc_curve(y_test, model_gnb.predict_proba(X_test)[:, 1], pos_label = 1)
fpr6, tpr6, thresh6 = roc_curve(y_test, model_XGB.predict_proba(X_test)[:, 1], pos_label = 1)

from sklearn.metrics import roc_auc_score
auc_score1 = roc_auc_score(y_test, model_LR.predict_proba(X_test)[:, 1])
auc_score2 = roc_auc_score(y_test, model_svm.predict_proba(X_test)[:, 1])
auc_score3 = roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1])
auc_score4 = roc_auc_score(y_test, model_knn.predict_proba(X_test)[:, 1])
auc_score5 = roc_auc_score(y_test, model_gnb.predict_proba(X_test)[:, 1])
auc_score6 = roc_auc_score(y_test, model_XGB.predict_proba(X_test)[:, 1])

print("XG Boost: ", auc_score6) # XGBoost
print("Logistic Regression: ", auc_score1) # Logistic Regression
print("Support Vector Machine: ", auc_score2) # Support Vector Machine
print("Random Forest: ", auc_score3) # Random Forest
print("K-Nearest Neighbors: ", auc_score4) # K-Nearest Neighbors
print("Naive Bayes: ", auc_score5) # Naive Bayes

plt.plot(fpr6, tpr6, linestyle = "--", color = "pink", label = "XGBoost")
plt.plot(fpr1, tpr1, linestyle = "--", color = "orange", label = "Logistic Regression")
plt.plot(fpr2, tpr2, linestyle = "--", color = "red", label = "SVM")
plt.plot(fpr3, tpr3, linestyle = "--", color = "green", label = "Random Forest")
plt.plot(fpr4, tpr4, linestyle = "--", color = "yellow", label = "KNN")
plt.plot(fpr5, tpr5, linestyle = "--", color = "blue", label = "Naive bayes")

plt.title('Receiver Operator Characteristics (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')

plt.legend(loc = 'best')
plt.savefig('ROC', dpi = 300)
plt.show()