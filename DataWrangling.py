import pandas as pd

pd.options.display.max_columns
pd.options.display.max_rows

hr_df = pd.read_csv('Human_Resources.csv')
#print(type(hr_df))

#To print first 6 rows
#print(hr_df.head(6))

#To print last 2 rows
#print(hr_df.tail(2))

#To print mean of each columns
#print(hr_df.mean(numeric_only=True))

#To get information about csv
#print(hr_df.info())
#print(hr_df.describe())

#To work on a particular column
#print(hr_df['Age'].mean())
#print(hr_df['Age'].max())
#print(hr_df['Age'].min())

#To see missing elements
#print(hr_df.isnull())

#total number of missing elements per column
#print(hr_df.isnull().sum())

# Drop any row that contains a Null value
#hr_df.dropna(how = 'any', inplace = True)

#print(hr_df)
# You can use Fillna to fill a given column with a certain value
#hr_df.fillna({'MonthlyIncome': hr_df['MonthlyIncome'].mean()}, inplace=True)
#print(hr_df)
#print(hr_df.isnull().sum())
#hr_df.fillna({'MonthlyRate': hr_df['MonthlyRate'].median()}, inplace=True)
#print(hr_df.isnull().sum())

#one hot encoding
hr_df = pd.read_csv('Human_Resources.csv')
print(hr_df['BusinessTravel'].unique()) #o/p : ['Travel_Rarely' 'Travel_Frequently' 'Non-Travel']
#print(hr_df['BusinessTravel'])

#now to create hot encoding:
BusinessTravel_Encoded=pd.get_dummies(hr_df['BusinessTravel'], dtype=int)
#print(BusinessTravel_Encoded)
hr_df.dropna(how = 'any', inplace = True)
print(hr_df['EducationField'].unique())
EducationField_Encoded=pd.get_dummies(hr_df['EducationField'], dtype=int)
print(EducationField_Encoded)