import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import seaborn as sns
import numpy as np

st.title("Market Data Visualization")

st.header('Introduction')
st.write("This application is designed to visualize and predict market data for major Health Care companies, specifically UnitedHealth Group (UNH), Moderna (MRNA), Eli Lilly (LLY), and Vertex Pharmaceuticals (VRTX).")
st.write("The data spans the last 5 years, providing insights into the performance of these companies in the health care sector.")
st.write("The app will display the historical stock prices, including open, high, low, close, adjusted close prices, and volume traded for each company.")
st.write("Users can explore the data through descriptive statistics and visualizations, helping them understand the trends and patterns in the health care market.")
st.write('Finally, the app will then look at using various machine learning techniques to predict the future stock prices of these companies.')


st.subheader('Dataset Overview')

#Creating the DataFrame
df = pd.DataFrame()
# Define ticker symbols for S&P 500 and FTSE 100
tickers = ["UNH", "MRNA", "LLY", "VRTX"]
#defining the start and end dates for the data
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
#Adding data to the DataFrame
df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, group_by='ticker')
df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index()
df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df.set_index('Date')

st.dataframe(df)

#Defining our Variables
st.subheader('Dataset Explanation')
st.write('Open: The price at which the index opened on that day.')
st.write('High: The highest price of the index during that day.')
st.write('Low: The lowest price of the index during that day.')
st.write('Close: The price at which the index closed on that day.')
st.write('Adj Close: The adjusted closing price, accounting for dividends and stock splits.')
st.write('Volume: The total number of shares traded during that day.')

#Descrive Statistics
df_description = df.describe()
st.subheader('Descriptive Statistics')
st.dataframe(df_description)


# Plotting the historical stock prices
st.subheader('Historical Stock Prices Overview')
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 7))
for ticker in tickers:
    ax.plot(df[df['Ticker'] == ticker]['Date'], df[df['Ticker'] == ticker]['Close'], label=ticker)
ax.set_title('Historical Stock Prices', fontsize=16, color='white')
ax.set_xlabel('Date', fontsize=14, color='white')
ax.set_ylabel('Stock Price (USD)', fontsize=14, color='white')
ax.tick_params(colors='white')
ax.legend(title='Ticker', fontsize=12, title_fontsize='13', loc='upper left')
ax.grid(alpha=0.3)
st.pyplot(fig)    









#Exploratory Data Analysis
st.header('Exploratory Data Analysis')

# Plotting the distribution of Open Prices for each company
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# UNH histogram
axes[0, 0].hist(df[df['Ticker'] == 'UNH']['Open'], bins=30, color= 'green', alpha=0.7, edgecolor='white')
axes[0, 0].set_title('Open Prices of UNH', fontsize=14, color='white')
axes[0, 0].set_xlabel('Open Price', fontsize=12, color='white')
axes[0, 0].set_ylabel('Frequency', fontsize=12, color='white')
axes[0, 0].tick_params(colors='white')
axes[0, 0].grid(alpha=0.3)

# MRNA histogram
axes[0, 1].hist(df[df['Ticker'] == 'MRNA']['Open'], bins=30, color= 'yellow', alpha=0.7, edgecolor='white')
axes[0, 1].set_title('Open Prices of MRNA', fontsize=14, color='white')
axes[0, 1].set_xlabel('Open Price', fontsize=12, color='white')
axes[0, 1].set_ylabel('Frequency', fontsize=12, color='white')
axes[0, 1].tick_params(colors='white')
axes[0, 1].grid(alpha=0.3)

# LLY histogram
axes[1, 0].hist(df[df['Ticker'] == 'LLY']['Open'], bins=30, color= 'blue', alpha=0.7, edgecolor='white')
axes[1, 0].set_title('Open Prices of LLY', fontsize=14, color='white')
axes[1, 0].set_xlabel('Open Price', fontsize=12, color='white')
axes[1, 0].set_ylabel('Frequency', fontsize=12, color='white')
axes[1, 0].tick_params(colors='white')
axes[1, 0].grid(alpha=0.3)

# LLY histogram
axes[1, 1].hist(df[df['Ticker'] == 'VRTX']['Open'], bins=30, color= 'red', alpha=0.7, edgecolor='white')
axes[1, 1].set_title('Open Prices of VRTX', fontsize=14, color='white')
axes[1, 1].set_xlabel('Open Price', fontsize=12, color='white')
axes[1, 1].set_ylabel('Frequency', fontsize=12, color='white')
axes[1, 1].tick_params(colors='white')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
st.subheader('Distribution of Open Prices')
st.pyplot(fig)
plt.close()

# Measuring the skewness of each company's Open Prices
st.subheader('Skewness of Open Prices')
skewness = df.groupby('Ticker')['Open'].apply(lambda x: x.skew())
print(skewness)
st.write(skewness)
st.write("Skewness indicates the asymmetry of the distribution of values. A skewness close to 0 suggests a symmetric distribution, while positive or negative values indicate right or left skewness, respectively.")
st.write('In the case of LLY, the skewness is 0.58, indicating a moderate right skew. This suggests that there are some higher open prices that are pulling the average up, but the distribution is not heavily skewed.')
st.write('For MRNA, the skewness is 1.47, indicating a significant right skew. This suggests that there are several high open prices that are pulling the average up, resulting in a distribution that is not symmetric.')
st.write('UNH has a skewness of -0.53, indicating a moderate left skew. This suggests that there are some lower open prices that are pulling the average down, but the distribution is not heavily skewed.')
st.write('VRTX has a skewness of 0.41, indicating a slight right skew. This suggests that there are some higher open prices that are pulling the average up, but the distribution is relatively symmetric overall.')
st.write('Overall, the skewness values suggest that the distributions of open prices for these companies are not perfectly symmetric, with some variations in the direction and degree of skewness.')
st.write('Typically, companies that are growing steadily might show left-skewed prices if major downturns (big drops) happen less often but are severe when they do (economic shocks, regulatory events). Companies with more volatility or speculative interest might show right-skewed prices if there are occasional spikes in price due to news or market sentiment, but the general trend is upward.')





# Plotting the distribution of Close Prices for each company
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# UNH histogram
axes[0, 0].hist(df[df['Ticker'] == 'UNH']['Close'], bins=30, color= 'green', alpha=0.7, edgecolor='white')
axes[0, 0].set_title('Close Prices of UNH', fontsize=14, color='white')
axes[0, 0].set_xlabel('Close Price', fontsize=12, color='white')
axes[0, 0].set_ylabel('Frequency', fontsize=12, color='white')
axes[0, 0].tick_params(colors='white')
axes[0, 0].grid(alpha=0.3)

# MRNA histogram
axes[0, 1].hist(df[df['Ticker'] == 'MRNA']['Close'], bins=30, color= 'yellow', alpha=0.7, edgecolor='white')
axes[0, 1].set_title('Close Prices of MRNA', fontsize=14, color='white')
axes[0, 1].set_xlabel('Close Price', fontsize=12, color='white')
axes[0, 1].set_ylabel('Frequency', fontsize=12, color='white')
axes[0, 1].tick_params(colors='white')
axes[0, 1].grid(alpha=0.3)

# LLY histogram
axes[1, 0].hist(df[df['Ticker'] == 'LLY']['Close'], bins=30, color= 'blue', alpha=0.7, edgecolor='white')
axes[1, 0].set_title('Close Prices of LLY', fontsize=14, color='white')
axes[1, 0].set_xlabel('Close Price', fontsize=12, color='white')
axes[1, 0].set_ylabel('Frequency', fontsize=12, color='white')
axes[1, 0].tick_params(colors='white')
axes[1, 0].grid(alpha=0.3)

# LLY histogram
axes[1, 1].hist(df[df['Ticker'] == 'VRTX']['Close'], bins=30, color= 'red', alpha=0.7, edgecolor='white')
axes[1, 1].set_title('Close Prices of VRTX', fontsize=14, color='white')
axes[1, 1].set_xlabel('Close Price', fontsize=12, color='white')
axes[1, 1].set_ylabel('Frequency', fontsize=12, color='white')
axes[1, 1].tick_params(colors='white')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
st.subheader('Distribution of Close Prices')
st.pyplot(fig)
plt.close()

# Measuring the skewness of each company's Open Prices
st.subheader('Skewness of Close Prices')
skewness = df.groupby('Ticker')['Close'].apply(lambda x: x.skew())
st.write(skewness)
st.write("Skewness indicates the asymmetry of the distribution of values. A skewness close to 0 suggests a symmetric distribution, while positive or negative values indicate right or left skewness, respectively.")
st.write('In the case of all companies, the skewness values for Close Prices are similar to those for Open Prices, indicating that the distributions of Close Prices are also not perfectly symmetric.')
st.write('LLY has a skewness of 0.57, indicating a moderate right skew. MRNA has a skewness of 1.47 indicating a significant right skew. UNH has a skewness of -0.53 indicating a moderate left skew. VRTX has a skewness of 0.41 indicating a slight right skew.')

# Plotting Open vs Close Prices for each company
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# UNH scatter plot
ax[0, 0].scatter(df[df['Ticker'] == 'UNH']['Open'], df[df['Ticker'] == 'UNH']['Close'], color = 'green',label=ticker, alpha=0.6)
ax[0, 0].set_title('UNH Open vs Close Prices', fontsize=16, color='white')
ax[0, 0].set_xlabel('Open Price (USD)', fontsize=14, color='white')
ax[0, 0].set_ylabel('Close Price (USD)', fontsize=14, color='white')
ax[0, 0].tick_params(colors='white')
ax[0, 0].grid(alpha=0.3)

# MRNA scatter plot
ax[0, 1].scatter(df[df['Ticker'] == 'MRNA']['Open'], df[df['Ticker'] == 'MRNA']['Close'], color = 'yellow',label=ticker, alpha=0.6)
ax[0, 1].set_title('MRNA Open vs Close Prices', fontsize=16, color='white')
ax[0, 1].set_xlabel('Open Price (USD)', fontsize=14, color='white')
ax[0, 1].set_ylabel('Close Price (USD)', fontsize=14, color='white')
ax[0, 1].tick_params(colors='white')
ax[0, 1].grid(alpha=0.3)

# LLY scatter plot
ax[1, 0].scatter(df[df['Ticker'] == 'LLY']['Open'], df[df['Ticker'] == 'LLY']['Close'], color = 'blue',label=ticker, alpha=0.6)
ax[1, 0].set_title('LLY Open vs Close Prices', fontsize=16, color='white')
ax[1, 0].set_xlabel('Open Price (USD)', fontsize=14, color='white')
ax[1, 0].set_ylabel('Close Price (USD)', fontsize=14, color='white')
ax[1, 0].tick_params(colors='white')
ax[1, 0].grid(alpha=0.3)

# VRTX scatter plot
ax[1, 1].scatter(df[df['Ticker'] == 'VRTX']['Open'], df[df['Ticker'] == 'VRTX']['Close'], color = 'red',label=ticker, alpha=0.6)
ax[1, 1].set_title('VRTX Open vs Close Prices', fontsize=16, color='white')
ax[1, 1].set_xlabel('Open Price (USD)', fontsize=14, color='white')
ax[1, 1].set_ylabel('Close Price (USD)', fontsize=14, color='white')
ax[1, 1].tick_params(colors='white')
ax[1, 1].grid(alpha=0.3)

plt.tight_layout()
st.subheader('Open vs Close Prices Scatter Plot')
st.pyplot(fig)
plt.close() 

st.write("The scatter plots show the relationship between Open and Close prices for each company. A positive correlation is observed, indicating that as the Open price increases, the Close price tends to increase as well. This is expected in financial markets, where the opening price often sets the tone for the day's trading.")
st.write("The Open vs Close prices for each company are similar, indicating minimal intraday price fluctuations. This suggests that the market sentiment for these companies is relatively stable, with prices not deviating significantly from their opening levels throughout the trading day.")


# Plotting Daily Returns
df['Daily Return'] = df.groupby('Ticker')['Adj Close'].pct_change()
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0, 0].plot(df[df['Ticker'] == 'UNH']['Date'], df[df['Ticker'] == 'UNH']['Daily Return'], color = 'green',label=ticker, alpha=0.6, marker='o', markersize=2)
ax[0, 0].set_title('UNH Daily Returns', fontsize=16, color='white')
ax[0, 0].set_xlabel('Date', fontsize=14, color='white')
ax[0, 0].set_ylabel('Daily Return', fontsize=14, color='white')
ax[0, 0].tick_params(colors='white')
ax[0, 0].grid(alpha=0.3)

ax[0, 1].plot(df[df['Ticker'] == 'MRNA']['Date'], df[df['Ticker'] == 'MRNA']['Daily Return'], color = 'yellow',label=ticker, alpha=0.6, marker='o', markersize=2)
ax[0, 1].set_title('MRNA Daily Returns', fontsize=16, color='white')
ax[0, 1].set_xlabel('Date', fontsize=14, color='white')
ax[0, 1].set_ylabel('Daily Return', fontsize=14, color='white')
ax[0, 1].tick_params(colors='white')
ax[0, 1].grid(alpha=0.3)    

ax[1, 0].plot(df[df['Ticker'] == 'LLY']['Date'], df[df['Ticker'] == 'LLY']['Daily Return'], color = 'blue',label=ticker, alpha=0.6, marker='o', markersize=2)
ax[1, 0].set_title('LLY Daily Returns', fontsize=16, color='white')
ax[1, 0].set_xlabel('Date', fontsize=14, color='white')
ax[1, 0].set_ylabel('Daily Return', fontsize=14, color='white')
ax[1, 0].tick_params(colors='white')
ax[1, 0].grid(alpha=0.3)    

ax[1, 1].plot(df[df['Ticker'] == 'VRTX']['Date'], df[df['Ticker'] == 'VRTX']['Daily Return'], color = 'red',label=ticker, alpha=0.6, marker='o', markersize=2)
ax[1, 1].set_title('VRTX Daily Returns', fontsize=16, color='white')
ax[1, 1].set_xlabel('Date', fontsize=14, color='white')
ax[1, 1].set_ylabel('Daily Return', fontsize=14, color='white')
ax[1, 1].tick_params(colors='white')
ax[1, 1].grid(alpha=0.3)

plt.tight_layout()
st.subheader('Daily Returns Plot')
st.pyplot(fig)
plt.close()

st.write("The daily returns plot shows the percentage change in the adjusted close price from one day to the next for each company. This provides insights into the volatility and performance of each stock over time.")   
st.write("The daily returns for all companies show fluctuations, with some days experiencing significant positive or negative returns. This indicates that the stocks are subject to market volatility, which is common in the health care sector due to various factors such as regulatory changes, clinical trial results, and market sentiment.")
st.write("The daily returns for MRNA show the highest volatility, with several days of significant positive and negative returns. This is likely due to the company's involvement in the COVID-19 vaccine development, which has led to rapid changes in stock price based on news and market sentiment.")
st.write("UNH, LLY, and VRTX show relatively stable daily returns compared to MRNA, indicating less volatility in their stock prices. This suggests that these companies have a more consistent performance in the market, which may be attributed to their established presence and steady growth in the health care sector.") 
st.write("Overall, the daily returns analysis provides valuable insights into the performance and volatility of these health care companies, helping investors make informed decisions based on historical trends and market behavior.")

st.subheader('Average Daily Returns')
average_daily_returns = df.groupby('Ticker')['Daily Return'].mean()
st.write(average_daily_returns)
st.write("The average daily returns provide insights into the overall performance of each stock over the analyzed period. A positive average daily return indicates that the stock has generally increased in value, while a negative average daily return suggests a decline in value.")
st.write("In this case, LLY has the highest average daily return at 0.0014, indicating a slight upward trend in its stock price over the analyzed period. MRNA follows closely with an average daily return of 0.0013, also suggesting a positive performance.")
st.write("VRTX has a lower average daily return of 0.0006, indicating a more stable performance with less volatility. VRTX has the lowest average daily return at 0.0002, suggesting that its stock price has been relatively stable with minimal fluctuations during the analyzed period.")

# Plotting Adjusted Close Prices for each company
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
ax[0, 0].plot(df[df['Ticker'] == 'UNH']['Date'], df[df['Ticker'] == 'UNH']['Adj Close'], color = 'green',label=ticker, alpha=0.6)
ax[0, 0].set_title('UNH Adjusted Close Prices', fontsize=16, color='white')
ax[0, 0].set_xlabel('Date', fontsize=14, color='white')
ax[0, 0].set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
ax[0, 0].tick_params(colors='white')
ax[0, 0].grid(alpha=0.3)

ax[0, 1].plot(df[df['Ticker'] == 'MRNA']['Date'], df[df['Ticker'] == 'MRNA']['Adj Close'], color = 'yellow',label=ticker, alpha=0.6)
ax[0, 1].set_title('MRNA Adjusted Close Prices', fontsize=16, color='white')
ax[0, 1].set_xlabel('Date', fontsize=14, color='white')
ax[0, 1].set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
ax[0, 1].tick_params(colors='white')
ax[0, 1].grid(alpha=0.3)

ax[1, 0].plot(df[df['Ticker'] == 'LLY']['Date'], df[df['Ticker'] == 'LLY']['Adj Close'], color = 'blue',label=ticker, alpha=0.6)
ax[1, 0].set_title('LLY Adjusted Close Prices', fontsize=16, color='white')
ax[1, 0].set_xlabel('Date', fontsize=14, color='white')
ax[1, 0].set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
ax[1, 0].tick_params(colors='white')
ax[1, 0].grid(alpha=0.3)    

ax[1, 1].plot(df[df['Ticker'] == 'VRTX']['Date'], df[df['Ticker'] == 'VRTX']['Adj Close'], color = 'red',label=ticker, alpha=0.6)
ax[1, 1].set_title('VRTX Adjusted Close Prices', fontsize=16, color='white')
ax[1, 1].set_xlabel('Date', fontsize=14, color='white')
ax[1, 1].set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
ax[1, 1].tick_params(colors='white')
ax[1, 1].grid(alpha=0.3)

plt.tight_layout()
st.subheader('Adjusted Close Prices Plot')
st.pyplot(fig)
plt.close()

st.write("The adjusted close prices plot shows the stock prices adjusted for dividends and stock splits, providing a more accurate representation of the stock's performance over time. This is particularly useful for long-term analysis, as it accounts for any corporate actions that may affect the stock price.")
st.write("The adjusted close prices for all companies vary over time, reflecting the overall trends in the health care sector. MRNA shows significant fluctuations, particularly during the COVID-19 pandemic, while UNH, LLY, and VRTX exhibit more stable price movements.")
st.write("The adjusted close prices for MRNA show the highest volatility, with significant spikes and drops in price. This is likely due to the company's involvement in the COVID-19 vaccine development, which has led to rapid changes in stock price based on news and market sentiment. The company's stock price saw an explosive increase during the early stages of the pandemic, followed a continuous decline since late 2021.")
st.write("UNH stock price has shown a mostly steady upward trend, reflecting the company's strong performance in the health care sector. The recent sharp fall in the stock price in early 2025 is likely due to a significant market correction or negative news affecting the company.")
st.write("LLY and VRTX also show strong upward trends, indicating their solid performance in the health care sector. Both companies have experienced steady growth over the analyzed period, with occasional fluctuations in their stock prices. The recent fall in both companies' stock prices in early 2025 is likely due to a significant market correction or negative news affecting the health care sector as a whole.")



# Calculating moving averages
df['MA_20'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=20).mean())
df['MA_50'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=50).mean())

# Plotting moving averages for each company
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
ax[0, 0].plot(df[df['Ticker'] == 'UNH']['Date'], df[df['Ticker'] == 'UNH']['Adj Close'], color = 'green', label='Adj Close', alpha=0.6)
ax[0, 0].plot(df[df['Ticker'] == 'UNH']['Date'], df[df['Ticker'] == 'UNH']['MA_20'], color='orange', label='MA 20', alpha=0.6)
ax[0, 0].plot(df[df['Ticker'] == 'UNH']['Date'], df[df['Ticker'] == 'UNH']['MA_50'], color='red', label='MA 50', alpha=0.6)
ax[0, 0].set_title('UNH Moving Averages', fontsize=16, color='white')
ax[0, 0].set_xlabel('Date', fontsize=14, color='white')
ax[0, 0].set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
ax[0, 0].tick_params(colors='white')
ax[0, 0].legend(title='Legend', fontsize=12, title_fontsize='13', loc='upper left')
ax[0, 0].grid(alpha=0.3)    

ax[0, 1].plot(df[df['Ticker'] == 'MRNA']['Date'], df[df['Ticker'] == 'MRNA']['Adj Close'], color = 'yellow', label='Adj Close', alpha=0.6)
ax[0, 1].plot(df[df['Ticker'] == 'MRNA']['Date'], df[df['Ticker'] == 'MRNA']['MA_20'], color='orange', label='MA 20', alpha=0.6)
ax[0, 1].plot(df[df['Ticker'] == 'MRNA']['Date'], df[df['Ticker'] == 'MRNA']['MA_50'], color='red', label='MA 50', alpha=0.6)
ax[0, 1].set_title('MRNA Moving Averages', fontsize=16, color='white')
ax[0, 1].set_xlabel('Date', fontsize=14, color='white')
ax[0, 1].set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
ax[0, 1].tick_params(colors='white')
ax[0, 1].legend(title='Legend', fontsize=12, title_fontsize='13', loc='upper left')
ax[0, 1].grid(alpha=0.3)    

ax[1, 0].plot(df[df['Ticker'] == 'LLY']['Date'], df[df['Ticker'] == 'LLY']['Adj Close'], color = 'blue', label='Adj Close', alpha=0.6)
ax[1, 0].plot(df[df['Ticker'] == 'LLY']['Date'], df[df['Ticker'] == 'LLY']['MA_20'], color='orange', label='MA 20', alpha=0.6)
ax[1, 0].plot(df[df['Ticker'] == 'LLY']['Date'], df[df['Ticker'] == 'LLY']['MA_50'], color='red', label='MA 50', alpha=0.6)
ax[1, 0].set_title('LLY Moving Averages', fontsize=16, color='white')
ax[1, 0].set_xlabel('Date', fontsize=14, color='white')
ax[1, 0].set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
ax[1, 0].tick_params(colors='white')
ax[1, 0].legend(title='Legend', fontsize=12, title_fontsize='13', loc='upper left')
ax[1, 0].grid(alpha=0.3)    

ax[1, 1].plot(df[df['Ticker'] == 'VRTX']['Date'], df[df['Ticker'] == 'VRTX']['Adj Close'], color = 'red', label='Adj Close', alpha=0.6)
ax[1, 1].plot(df[df['Ticker'] == 'VRTX']['Date'], df[df['Ticker'] == 'VRTX']['MA_20'], color='orange', label='MA 20', alpha=0.6)
ax[1, 1].plot(df[df['Ticker'] == 'VRTX']['Date'], df[df['Ticker'] == 'VRTX']['MA_50'], color='red', label='MA 50', alpha=0.6)
ax[1, 1].set_title('VRTX Moving Averages', fontsize=16, color='white')
ax[1, 1].set_xlabel('Date', fontsize=14, color='white')
ax[1, 1].set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
ax[1, 1].tick_params(colors='white')
ax[1, 1].legend(title='Legend', fontsize=12, title_fontsize='13', loc='upper left')
ax[1, 1].grid(alpha=0.3)    

plt.tight_layout()
st.subheader('Moving Averages Plot')
st.pyplot(fig)
plt.close()

st.write("The moving averages plot shows the 20-day and 50-day moving averages of the adjusted close prices for each company. Moving averages are commonly used in technical analysis to smooth out price data and identify trends over time.")
st.write("The 20-day moving average is more responsive to recent price changes, while the 50-day moving average provides a longer-term perspective. The crossover of these moving averages can indicate potential buy or sell signals.")



st.subheader('Correlation between Companies Closing Prices')
correlation = df.pivot(index='Date', columns='Ticker', values='Close').corr()
st.write(correlation)

# Plotting the correlation heatmap
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Correlation Heatmap of Closing Prices', fontsize=16, color='white')
ax.tick_params(colors='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12, color='white')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, color='white')
plt.tight_layout()
st.subheader('Correlation Heatmap')
st.pyplot(fig)
plt.close()

st.write("The correlation matrix shows the relationship between the closing prices of the companies. A value close to 1 indicates a strong positive correlation, meaning that the prices tend to move in the same direction. A value close to -1 indicates a strong negative correlation, meaning that the prices tend to move in opposite directions. A value close to 0 indicates no correlation.")
st.write("In this case, companies show a variety of correlations, with some companies moving closely together while others show more independent price movements.")
st.write("The highest correlation is between LLY and VRTX (0.95), indicating that these two companies tend to move in the same direction very closely. This could be due to their similar business models and market conditions.")
st.write("The next highest correlations are between UNH and LLY (0.54), and UNH and VRTX (0.46), suggesting that while both these pairs are all in the health care sector, their stock prices do not move as closely together as LLY and VRTX.") 
st.write("The lowest correlations are between MRNA and VRTX (-0.55), and MRNA and LLY (-0.35), indicating that these companies' stock prices tend to move in opposite directions at times. This could be due to MRNA's focus on mRNA technology and vaccines, which may not be directly influenced by the same factors affecting LLY and VRTX, which are more focused on traditional pharmaceuticals and treatments.")
st.write("Overall, the correlation analysis provides insights into the relationships between the stock prices of these health care companies, helping investors understand how they may move together in response to market trends and economic conditions.")




adj_close = df.pivot_table(values = 'Daily Return', index = 'Date', columns = 'Ticker')
# Creating a pairplot of adjusted close prices
plt.style.use('dark_background')
sns.set(style='darkgrid')
pairplot = sns.PairGrid(adj_close)
pairplot.map_upper(plt.scatter, color='red', alpha=0.6)
pairplot.map_lower(sns.kdeplot, color='green', alpha=0.6)
pairplot.map_diag(plt.hist, bins = 30, color='blue', alpha=0.6)
pairplot.fig.suptitle('Pairplot of Daily Returns', fontsize=16, color='white')
pairplot.fig.tight_layout()
st.subheader('Pairplot of Daily Returns')
st.pyplot(pairplot.fig)
plt.close()

st.write("The pairplot of daily returns provides a visual representation of the relationships between the daily returns of the companies. The scatter plots in the upper triangle show the correlation between pairs of companies, while the lower triangle shows the distribution of daily returns for each company.")
st.write("The diagonal histograms show the distribution of daily returns for each company, indicating how the returns are distributed over time. The KDE plots in the lower triangle provide a smoothed estimate of the distribution of daily returns, helping to visualize the density of returns.")


st.subheader('Volatility of Each Company')
plt.style.use('dark_background')
rets = adj_close.dropna()
area = np.pi * 20
plt.figure(figsize=(10, 6))
plt.scatter(rets.mean(), rets.std(), s=area, alpha=0.5)
plt.title('Volatility of Each Company', fontsize=16, color='white')
plt.xlabel('Expected Return', fontsize=14, color='white')
plt.ylabel('Risk', fontsize=14, color='white')
plt.tick_params(colors='white')
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha = 'right', va = 'bottom', fontsize=12, color='white', arrowprops=dict(arrowstyle='-', color='red', connectionstyle = 'arc3,rad=0.3'))
plt.grid(alpha=0.3)
st.pyplot(plt.gcf())
plt.close()

st.write("The volatility plot shows the relationship between the mean daily return and the standard deviation of daily returns for each company. The size of the points represents the average daily return, while the position of the points indicates the volatility (standard deviation) of the daily returns.")
