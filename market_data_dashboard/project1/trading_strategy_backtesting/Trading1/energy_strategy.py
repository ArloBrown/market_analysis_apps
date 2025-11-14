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


plt.style.use('dark_background')
plt.grid(alpha=0.3)

sns.set_context("talk") 
plt.rcParams["figure.facecolor"] = "#0e1117" 
plt.rcParams["axes.facecolor"] = "#0e1117"    
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"

st.set_page_config(page_title="Energy Commodity Strategy Analysis", layout="wide")
st.title("Energy Commodity Strategy Analysis (WTI Crude Oil)")



# Options: "CL=F" (WTI), "BZ=F" (Brent), "NG=F" (Nat Gas)
ticker = "CL=F"  
df = yf.download(ticker, start="2015-01-01", interval="1d")
df.dropna(inplace=True)

st.write(f"### {ticker} Data Overview (Crude Oil)")
st.dataframe(df.head())


# RETURNS

df['Return'] = df['Close'].pct_change()
df.dropna(inplace=True)

st.write("### Daily Returns Overview")
st.dataframe(df[['Close', 'Return']].head())


# TREND ANALYSIS (Moving Averages)

df['SMA50'] = df['Close'].rolling(50).mean()
df['SMA200'] = df['Close'].rolling(200).mean()

st.write("### SMA Trend Analysis")
st.dataframe(df[['Close', 'SMA50', 'SMA200']].tail())

plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='WTI Crude Oil', alpha=0.8)
plt.plot(df['SMA50'], label='SMA 50')
plt.plot(df['SMA200'], label='SMA 200')
plt.title('WTI Crude Oil Prices with SMAs')
plt.legend()
st.pyplot(plt.gcf())
plt.close()


# VOLATILITY (ATR)

df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

plt.figure(figsize=(12,4))
plt.plot(df['ATR'], label='ATR(14)')
plt.title('Average True Range (Volatility)')
plt.legend()
st.pyplot(plt.gcf())
plt.close()

st.write("### ATR (Volatility) Overview")
st.dataframe(df[['Close', 'ATR']].tail())


# RETURN DISTRIBUTION

plt.figure(figsize=(8,4))
sns.histplot(df['Return'], bins=50, kde=True)
plt.title('WTI Daily Return Distribution')
st.pyplot(plt.gcf())
plt.close()

st.write(f"**Return Skewness:** {df['Return'].skew():.4f}")
st.write(f"**Return Kurtosis:** {df['Return'].kurtosis():.4f}")


# AUTOCORRELATION

from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(8,4))
autocorrelation_plot(df['Return'])
plt.title('Autocorrelation of WTI Returns')
st.pyplot(plt.gcf())
plt.close()


# STATIONARITY TEST

adf_result = adfuller(df['Close'])

st.write("### ADF Stationarity Test")
st.write(f"ADF Statistic: {adf_result[0]}")
st.write(f"p-value: {adf_result[1]}")
st.write("Series is **Stationary**" if adf_result[1] < 0.05 else "Series is **NOT Stationary**")


# DAY-OF-WEEK PATTERNS

df['DayOfWeek'] = df.index.day_name()
dow_stats = df.groupby('DayOfWeek')['Return'].mean().sort_values()

plt.figure(figsize=(8,4))
dow_stats.plot(kind='bar')
plt.title('Average Return by Day of Week (WTI)')
st.pyplot(plt.gcf())
plt.close()


# CORRELATION WITH OTHER ENERGY ASSETS


# Natural Gas (NG=F)
natgas = yf.download("NG=F", start="2015-01-01", interval="1d")['Close']
natgas.name = "NaturalGas"

# Brent Crude (BZ=F)
brent = yf.download("BZ=F", start="2015-01-01", interval="1d")['Close']
brent.name = "Brent"

# Energy Sector ETF (XLE)
xle = yf.download("XLE", start="2015-01-01", interval="1d")['Close']
xle.name = "XLE"

wti = df['Close'].copy()
wti.name = "WTI"

corr_df = pd.concat([wti, brent, natgas, xle], axis=1).dropna()

st.write("### Correlation Data Overview")
st.dataframe(corr_df.head())

plt.figure(figsize=(8,6))
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap (Energy Assets)')
st.pyplot(plt.gcf())
plt.close()
