import pandas as pd

hr_df = pd.read_csv('Human_Resources.csv')

#print(hr_df.info())

# It makes sense to drop 'EmployeeCount', 'Standardhours' and 'Over18' since they do not change from one employee to the other
# Let's drop 'EmployeeNumber' as well, axis=1 means columns, 0 means rows
hr_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)
print(hr_df.info())

#Let's see how many employees left the company
left_df = hr_df[hr_df['Attrition'] == 'Yes']
print(left_df.describe()) #gives statistical summary

#Employees who stayed in the company
stayed_df = hr_df[hr_df['Attrition'] == 'No']
print(stayed_df.describe())