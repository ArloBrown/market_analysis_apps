import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import seaborn as sns
import numpy as np

st.title("Energy Sector Market Data Visualization")

st.header('Introduction')
st.write("""
This application is designed to visualize and analyze market data for major global Energy companies,
specifically Exxon Mobil (XOM), Chevron (CVX), Shell PLC (SHEL), and BP PLC (BP).
""")
st.write("""
The dataset spans the past 5 years and provides insights into the performance of these companies in the energy sector,
including traditional oil & gas exploration, refining, and integrated energy operations.
""")
st.write("""
The app displays historical stock prices such as open, high, low, close, adjusted close, and trading volume.  
You can explore descriptive statistics, trends, and patterns that define the energy market—for example,
commodity price fluctuations, geopolitical risk, OPEC decisions, and demand/supply cycles.
""")
st.write("""
Finally, the application includes analytical and visualization tools that help reveal volatility, correlation,
and other characteristics useful for investors studying the energy sector.
""")

# --------------------------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------------------------
st.subheader('Dataset Overview')

tickers = ["XOM", "CVX", "SHEL", "BP"]

start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, group_by='ticker')
df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index()
df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df.set_index('Date')

st.dataframe(df)

# --------------------------------------------------------------------
# VARIABLE EXPLANATION
# --------------------------------------------------------------------
st.subheader('Dataset Explanation')
st.write('Open: The price at which the stock opened on that day.')
st.write('High: The highest price of the stock during that day.')
st.write('Low: The lowest price of the stock during that day.')
st.write('Close: The price at which the stock closed at market close.')
st.write('Adj Close: Adjusted closing price with dividends and splits applied.')
st.write('Volume: The total number of shares traded during the day.')

# --------------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# --------------------------------------------------------------------
df_description = df.describe()
st.subheader('Descriptive Statistics')
st.dataframe(df_description)

# --------------------------------------------------------------------
# HISTORICAL PRICE PLOT
# --------------------------------------------------------------------
st.subheader('Historical Stock Prices Overview')
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 7))
for ticker in tickers:
    ax.plot(df[df['Ticker'] == ticker]['Date'], df[df['Ticker'] == ticker]['Close'], label=ticker)
ax.set_title('Historical Stock Prices (Energy Sector)', fontsize=16, color='white')
ax.set_xlabel('Date', fontsize=14, color='white')
ax.set_ylabel('Stock Price (USD)', fontsize=14, color='white')
ax.tick_params(colors='white')
ax.legend(title='Ticker', fontsize=12, title_fontsize='13', loc='upper left')
ax.grid(alpha=0.3)
st.pyplot(fig)

# --------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------
st.header('Exploratory Data Analysis')

# ----------------------------
# DISTRIBUTION OF OPEN PRICES
# ----------------------------
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

company_titles = {
    "XOM": "Exxon Mobil (XOM)",
    "CVX": "Chevron (CVX)",
    "SHEL": "Shell PLC (SHEL)",
    "BP": "BP PLC (BP)"
}

colors = ["green", "yellow", "blue", "red"]

for ax, ticker, color in zip(axes.flatten(), tickers, colors):
    ax.hist(df[df['Ticker'] == ticker]['Open'], bins=30, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(f'Open Prices of {ticker}', fontsize=14, color='white')
    ax.set_xlabel('Open Price', fontsize=12, color='white')
    ax.set_ylabel('Frequency', fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.3)

plt.tight_layout()
st.subheader('Distribution of Open Prices')
st.pyplot(fig)
plt.close()

# ----------------------------
# SKEWNESS (OPEN PRICES)
# ----------------------------
st.subheader('Skewness of Open Prices')
skewness = df.groupby('Ticker')['Open'].apply(lambda x: x.skew())
st.write(skewness)
st.write("""
Skewness measures asymmetry in the distribution of stock prices.
Energy companies often show skewness due to oil price shocks,
geopolitical tensions, or supply/demand imbalances.
""")

# ----------------------------
# DISTRIBUTION OF CLOSE PRICES
# ----------------------------
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, ticker, color in zip(axes.flatten(), tickers, colors):
    ax.hist(df[df['Ticker'] == ticker]['Close'], bins=30, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(f'Close Prices of {ticker}', fontsize=14, color='white')
    ax.set_xlabel('Close Price', fontsize=12, color='white')
    ax.set_ylabel('Frequency', fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.3)

plt.tight_layout()
st.subheader('Distribution of Close Prices')
st.pyplot(fig)
plt.close()

st.subheader('Skewness of Close Prices')
st.write(df.groupby('Ticker')['Close'].apply(lambda x: x.skew()))

# ----------------------------
# OPEN VS CLOSE SCATTER
# ----------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

for subplot, ticker, color in zip(ax.flatten(), tickers, colors):
    subplot.scatter(
        df[df['Ticker'] == ticker]['Open'],
        df[df['Ticker'] == ticker]['Close'],
        color=color, alpha=0.6
    )
    subplot.set_title(f'{ticker} Open vs Close Prices', fontsize=16, color='white')
    subplot.set_xlabel('Open Price (USD)', fontsize=14, color='white')
    subplot.set_ylabel('Close Price (USD)', fontsize=14, color='white')
    subplot.tick_params(colors='white')
    subplot.grid(alpha=0.3)

plt.tight_layout()
st.subheader('Open vs Close Prices Scatter Plot')
st.pyplot(fig)
plt.close()

# --------------------------------------------------------------------
# DAILY RETURNS
# --------------------------------------------------------------------
df['Daily Return'] = df.groupby('Ticker')['Adj Close'].pct_change()

plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

for subplot, ticker, color in zip(ax.flatten(), tickers, colors):
    subset = df[df['Ticker'] == ticker]
    subplot.plot(
        subset['Date'], subset['Daily Return'],
        color=color, alpha=0.6, marker='o', markersize=2
    )
    subplot.set_title(f'{ticker} Daily Returns', fontsize=16, color='white')
    subplot.set_xlabel('Date', fontsize=14, color='white')
    subplot.set_ylabel('Daily Return', fontsize=14, color='white')
    subplot.tick_params(colors='white')
    subplot.grid(alpha=0.3)

plt.tight_layout()
st.subheader('Daily Returns Plot')
st.pyplot(fig)
plt.close()

st.subheader('Average Daily Returns')
average_daily_returns = df.groupby('Ticker')['Daily Return'].mean()
st.write(average_daily_returns)

# --------------------------------------------------------------------
# ADJUSTED CLOSE PRICES
# --------------------------------------------------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

for subplot, ticker, color in zip(ax.flatten(), tickers, colors):
    subset = df[df['Ticker'] == ticker]
    subplot.plot(subset['Date'], subset['Adj Close'], color=color, alpha=0.6)
    subplot.set_title(f'{ticker} Adjusted Close Prices', fontsize=16, color='white')
    subplot.set_xlabel('Date', fontsize=14, color='white')
    subplot.set_ylabel('Adjusted Close Price (USD)', fontsize=14, color='white')
    subplot.tick_params(colors='white')
    subplot.grid(alpha=0.3)

plt.tight_layout()
st.subheader('Adjusted Close Prices Plot')
st.pyplot(fig)
plt.close()

# --------------------------------------------------------------------
# MOVING AVERAGES
# --------------------------------------------------------------------
df['MA_20'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=20).mean())
df['MA_50'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=50).mean())

plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

for subplot, ticker, color in zip(ax.flatten(), tickers, colors):
    subset = df[df['Ticker'] == ticker]
    subplot.plot(subset['Date'], subset['Adj Close'], color=color, label='Adj Close', alpha=0.6)
    subplot.plot(subset['Date'], subset['MA_20'], color='orange', label='MA 20', alpha=0.6)
    subplot.plot(subset['Date'], subset['MA_50'], color='red', label='MA 50', alpha=0.6)
    subplot.set_title(f'{ticker} Moving Averages', fontsize=16, color='white')
    subplot.tick_params(colors='white')
    subplot.legend()
    subplot.grid(alpha=0.3)

plt.tight_layout()
st.subheader('Moving Averages Plot')
st.pyplot(fig)
plt.close()

# --------------------------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------------------------
st.subheader('Correlation between Companies Closing Prices')
correlation = df.pivot(index='Date', columns='Ticker', values='Close').corr()
st.write(correlation)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Correlation Heatmap of Closing Prices (Energy Sector)', fontsize=16, color='white')
ax.tick_params(colors='white')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# --------------------------------------------------------------------
# PAIRPLOT OF DAILY RETURNS
# --------------------------------------------------------------------
adj_close = df.pivot_table(values='Daily Return', index='Date', columns='Ticker')

plt.style.use('dark_background')
sns.set(style='darkgrid')
pairplot = sns.PairGrid(adj_close)
pairplot.map_upper(plt.scatter, color='red', alpha=0.6)
pairplot.map_lower(sns.kdeplot, color='green', alpha=0.6)
pairplot.map_diag(plt.hist, bins=30, color='blue', alpha=0.6)
pairplot.fig.suptitle('Pairplot of Daily Returns', fontsize=16, color='white')
pairplot.fig.tight_layout()
st.subheader('Pairplot of Daily Returns')
st.pyplot(pairplot.fig)
plt.close()

# --------------------------------------------------------------------
# VOLATILITY PLOT
# --------------------------------------------------------------------
st.subheader('Volatility of Each Company')
rets = adj_close.dropna()
plt.figure(figsize=(10, 6))
plt.scatter(rets.mean(), rets.std(), s=np.pi * 20, alpha=0.5)
plt.title('Volatility of Each Energy Company', fontsize=16, color='white')
plt.xlabel('Expected Return', fontsize=14, color='white')
plt.ylabel('Risk (Std Dev of Returns)', fontsize=14, color='white')
plt.tick_params(colors='white')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, xy=(x, y), xytext=(50, 50), textcoords='offset points',
        ha='right', va='bottom', fontsize=12, color='white',
        arrowprops=dict(arrowstyle='-', color='red', connectionstyle='arc3,rad=0.3')
    )

plt.grid(alpha=0.3)
st.pyplot(plt.gcf())
plt.close()
