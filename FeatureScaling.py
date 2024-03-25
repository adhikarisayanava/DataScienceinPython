import pandas as pd

hr_df = pd.read_csv('Human_Resources.csv')
print(hr_df['Age'].values)

# Normalization is conducted to make feature values range from 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
hr_df['Age'] = scaler.fit_transform(hr_df['Age'].values.reshape(-1,1))
print(hr_df['Age'])
print(hr_df.describe())