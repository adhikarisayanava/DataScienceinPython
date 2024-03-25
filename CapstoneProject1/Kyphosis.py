import pandas as pd

def age_month_to_year(age_in_months):
    return age_in_months/12

kyphosis_df = pd.read_csv('kyphosis.csv')

#print(kyphosis_df.head(10))
#print(kyphosis_df.tail())
#print(kyphosis_df.describe())

print((kyphosis_df['Age'].min()/12).round(2))
print((kyphosis_df['Age'].max()/12).round(2))
print((kyphosis_df['Age'].mean()/12).round(2))

print(kyphosis_df.isnull().sum())

#converting the age column datatype from int64 to float64
kyphosis_df['Age'] = kyphosis_df['Age'].astype('float64')
print(kyphosis_df.info())

kyphosis_df['Age in Years'] = kyphosis_df['Age'].apply(age_month_to_year)
print(kyphosis_df)
print("----------------------------------------------------------")
print (kyphosis_df[kyphosis_df['Age'] == kyphosis_df['Age'].min()])
print("----------------------------------------------------------")
print (kyphosis_df[kyphosis_df['Age'] == kyphosis_df['Age'].max()])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
kyphosis_df = pd.read_csv('kyphosis.csv')
kyphosis_df['Age'] = scaler.fit_transform(kyphosis_df['Age'].values.reshape(-1,1))
print(kyphosis_df['Age'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
kyphosis_df = pd.read_csv('kyphosis.csv')
kyphosis_df['Age'] = scaler.fit_transform(kyphosis_df['Age'].values.reshape(-1,1))
print(kyphosis_df['Age'])
