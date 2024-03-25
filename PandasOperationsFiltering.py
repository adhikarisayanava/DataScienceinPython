import pandas as pd

hr_df = pd.read_csv('Human_Resources.csv')

loyal_employees_df = hr_df[ (hr_df['YearsAtCompany'] >= 30) ]
#print(loyal_employees_df)

mask_1 = hr_df['YearsAtCompany'] >= 30
mask_2 = hr_df['Department'] == 'Research & Development'

# Pick certain rows that satisfy 2 or more critirea
loyal_rnd_df = hr_df[mask_1 & mask_2]
#print(loyal_rnd_df)

# values that fall between a given range
#print(hr_df[hr_df["DailyRate"].between(800, 850)])

print(hr_df[hr_df['DailyRate'] >= 1450])

mask_1 = hr_df[hr_df['DailyRate'] >= 1450]
print(mask_1["DailyRate"].sum())


