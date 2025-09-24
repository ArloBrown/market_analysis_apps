import pandas as pd 
import numpy as np
import streamlit as st
import backtrader as bt
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
# Set dark theme for all plots
plt.style.use('dark_background')
plt.grid(alpha=0.3)


# Larger fonts, cleaner layout
sns.set_context("talk")  # options: paper, notebook, talk, poster
plt.rcParams["figure.facecolor"] = "#0e1117"  # Streamlit dark theme bg
plt.rcParams["axes.facecolor"] = "#0e1117"    # match Streamlit bg
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"



st.set_page_config(page_title="EUR/USD Strategy Analysis", layout="wide")
st.title("EUR/USD Strategy Analysis")


#Download EUR/USD data
ticker = "EURUSD=X"
df = yf.download(ticker, start="2015-01-01", interval="1d")
df.drop(columns=["Volume"], inplace=True)  # Volume is useless for FX
df.dropna(inplace=True)
print(df.head())
st.write("EUR/USD Data Overview")
st.dataframe(df.head())


#Calculate Returns
df['Return'] = df['Close'].pct_change() 
df.dropna(inplace=True)
st.write("EUR/USD Returns Overview")
st.dataframe(df[['Close', 'Return']].head())    


#Trend Analysis
df['SMA50'] = df['Close'].rolling(50).mean()
df['SMA200'] = df['Close'].rolling(200).mean()
st.write("SMA Analysis")
st.dataframe(df[['Close', 'SMA50', 'SMA200']].tail())

plt.style.use('dark_background')
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='EUR/USD', alpha=0.8)
plt.plot(df['SMA50'], label='SMA 50')
plt.plot(df['SMA200'], label='SMA 200')
plt.title('EUR/USD Price with SMAs')
plt.legend()
st.pyplot(plt.gcf())
plt.close()

# Volatility Analysis 

df['ATR'] = df['High'] - df['Low']
df['ATR'] = df['ATR'].rolling(14).mean()

plt.figure(figsize=(12,4))
plt.plot(df['ATR'], label='ATR(14)')
plt.title('Average True Range (Volatility)')
plt.legend()
plt.show()
st.write("Average True Range (ATR) Overview")
st.dataframe(df[['Close', 'ATR']].tail())
st.pyplot(plt.gcf())
plt.close()

#Return Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['Return'], bins=50, kde=True)
plt.title('EUR/USD Daily Return Distribution')
plt.show()

print("Return Skewness:", df['Return'].skew())
print("Return Kurtosis:", df['Return'].kurtosis())

st.write("Return Distribution Overview")
st.pyplot(plt.gcf())
plt.close() 


#Autocorrelation
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(8,4))
autocorrelation_plot(df['Return'])
plt.title('Autocorrelation of Returns')
plt.show()
st.write("Autocorrelation of Returns Overview")
st.pyplot(plt.gcf())
plt.close() 

#Stationarity Test
adf_result = adfuller(df['Close'])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
if adf_result[1] < 0.05:
    print("Series is stationary")
else:
    print("Series is NOT stationary")

st.write("ADF Test Result")
st.write(f"ADF Statistic: {adf_result[0]}")
st.write(f"p-value: {adf_result[1]}")   



# 8. Time-of-Day & Day-of-Week Patterns
# (Using daily data, we can only check day-of-week)
df['DayOfWeek'] = df.index.day_name()
dow_stats = df.groupby('DayOfWeek')['Return'].mean().sort_values()

plt.figure(figsize=(8,4))
dow_stats.plot(kind='bar')
plt.title('Average Return by Day of Week')
plt.show()
st.write("Average Return by Day of Week")
st.pyplot(plt.gcf())
plt.close() 

#Correlation with DXY & Gold
dxy = yf.download("DX-Y.NYB", start="2015-01-01", interval="1d")['Close']
gold = yf.download("GC=F", start="2015-01-01", interval="1d")['Close']

# Make sure each is a Series with a proper name
eurusd = df['Close'].copy()
eurusd.name = "EURUSD"

dxy = yf.download("DX-Y.NYB", start="2015-01-01", interval="1d")['Close'].copy()
dxy.name = "DXY"

gold = yf.download("GC=F", start="2015-01-01", interval="1d")['Close'].copy()
gold.name = "Gold"

# Concatenate on columns
corr_df = pd.concat([eurusd, dxy, gold], axis=1).dropna()

st.write("Correlation Data Overview")
st.dataframe(corr_df.head())    


print("Correlation Matrix:")
print(corr_df.corr())

sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()
st.write("Correlation Heatmap Overview")
st.pyplot(plt.gcf())
plt.close()

