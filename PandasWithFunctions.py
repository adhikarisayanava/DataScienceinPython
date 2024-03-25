import pandas as pd

def dailyrate_update(balance):
    return balance * 1.1 #daily rate increased by 10%


hr_df = pd.read_csv('Human_Resources.csv')
hr_df['DailyRate'] = hr_df['DailyRate'].apply(dailyrate_update)
print(hr_df)
