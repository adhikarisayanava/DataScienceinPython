import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

stocks_df = pd.read_csv('stocks_daily_prices.csv')
daily_return_df = pd.read_csv('stocks_daily_returns.csv')

#print(stocks_df.info())
#print(daily_return_df.info())

#Using Matplotlib, plot lineplots that display all 4 stocks daily prices on one single figure.
stocks_df.plot(x='Date',y = ['AAPL', 'JPM', 'PG', 'UAL'],linewidth=3, figsize=(20,12))
plt.ylabel('Prices')
plt.title('Stock Daily Prices')
plt.legend(loc= 'upper center')
plt.grid()
plt.show()

#Using Matplotlib, plot 4 stocks daily prices on multiple subplots.
plt.figure(figsize = (20, 4))

plt.subplot(1, 4, 1)
plt.plot(stocks_df['AAPL'], 'r--');
plt.grid()

plt.subplot(1, 4, 2)
plt.plot(stocks_df['JPM'], 'b.');
plt.grid()


plt.subplot(1, 4, 3)
plt.plot(stocks_df['PG'], 'g.');
plt.grid()

plt.subplot(1, 4, 4)
plt.plot(stocks_df['UAL'], 'y.');
plt.grid()
plt.show()


#Using Matplotlib, plot the scatterplot between Apple and JP Morgan daily returns. 
daily_return_df.plot.scatter('AAPL','JPM', grid=True, figsize=(10,5), color = 'hotpink', alpha =0.8)
plt.title('My First Scatter Plot!')
plt.show()

#Using Seaborn, plot similar scatterplot between Apple and JP Morgan daily returns. 
#plt.figure(figsize=(20,12))
sns.scatterplot(x="AAPL", y="JPM", data=daily_return_df)
plt.grid()
plt.show()


#Assume that you decided to become bullish on AAPL and you allocated 70% of your assets in it. You also decided to equally divide the rest of your assets in other stocks (JPM, PG, and UAL). Using Matplotlib, plot a pie chart that shows these allocations. Use 'explode’ attribute to increase the separation between AAPL and the rest of the portfolio.
allocation_df = pd.DataFrame(data = {'allocation %':[70, 10, 10, 10]}, index=['AAPL', 'JPM', 'PG', 'UAL'])
#print(crypto_df)
explode = (0.2,0,0,0)

allocation_df.plot.pie(y = 'allocation %', explode = explode)
plt.title('PIE CHART')
plt.show()


#Using Matplotlib, plot the histogram for United Airlines and P&G returns using 40 bins with red color. Display the mean and Standard deviation for both stocks on top of the figure.
mu_UAL = round(daily_return_df['UAL'].mean(), 2)
sigma_UAL = round(daily_return_df['UAL'].std(), 2)

mu_PG = round(daily_return_df['UAL'].mean(), 2)
sigma_PG = round(daily_return_df['UAL'].std(), 2)

num_bins = 40
#plt.figure(figsize=(10,5))

daily_return_df[['UAL', 'PG']].plot.hist(alpha = 0.7, figsize=(10,5), bins=num_bins, color='r')
plt.grid()
plt.ylabel('Probability')
plt.title('Histogram: UAL mu=' + str(mu_UAL) + ',sigma='+ str(sigma_UAL) + ', PG mu=' + str(mu_PG) + ',sigma='+ str(sigma_PG))
plt.show()

#Using Seaborn, plot a heatmap that shows the correlations between stocks daily returns. Comment on the correlation between UAL and P&G.
sns.heatmap(daily_return_df.corr(numeric_only=True), annot=True)
plt.show()


#Plot a 3D plot showing all daily returns from JPM, AAPL and UAL
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('JP Morgan')
ax.set_ylabel('Apple')
ax.set_zlabel('United Airlines')
x=daily_return_df['JPM']
y=daily_return_df['AAPL']
z=daily_return_df['UAL']
ax.scatter(x,y,z)
plt.show()

