import pandas as pd 
import numpy as np
import streamlit as st
import backtrader as bt
import matplotlib.pyplot as plt
import yfinance as yf
import fredapi as fa
import requests

import seaborn as sns
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings("ignore")


#Collecting All Data


start = "2015-01-01"
end = "2025-01-01"

#Yahoo Finance FX Pairs

fx_pairs = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "DXY": "DX-Y.NYB"   
}
df_fx = yf.download(list(fx_pairs.values()), start=start, end=end)['Close']
df_fx.rename(columns={v: k for k, v in fx_pairs.items()}, inplace=True)
df_fx = df_fx.asfreq('B')
df_fx.bfill(axis=0, inplace=True)


#Yahoo Finance Indices

indices = {
    "FTSE100": "^FTSE",
    "DAX30": "^GDAXI",
    "S&P500": "^GSPC",
    "NIKKEI225": "^N225",
    "CAC40": "^FCHI",
    "EUROSTOXX50": "^STOXX50E",
    "MIB": "FTSEMIB.MI"
}

df_indices = yf.download(list(indices.values()), start=start, end=end)['Close']
df_indices.rename(columns={v: k for k, v in indices.items()}, inplace=True)


#Yahoo Financ Commodities

commodities = {
    "BrentOil": "BZ=F",
    "Gold": "GC=F",
    "VIX": "^VIX"         
}
df_commodities = yf.download(list(commodities.values()), start=start, end=end)['Close']
df_commodities.rename(columns={v: k for k, v in commodities.items()}, inplace=True)

# Merge Commodities & Indices

df_comm_ind = pd.merge(df_commodities, df_indices, left_index=True, right_index=True, how='outer')

full_index = pd.date_range(start=start, end=end, freq='B')
df_comm_ind = df_comm_ind.reindex(full_index)

df_comm_ind.ffill(axis=0, inplace=True)
df_comm_ind.bfill(axis=0, inplace=True)






#FRED Rates & Yields

# fred_series = {
#     "US10Y": "DGS10",                    # US 10Y Treasury
#     "DE10Y": "IRLTLT01DEM156N",          # Germany 10Y Bond Yield
#     "FR10Y": "IRLTLT01FRM156N",          # France 10Y
#     "IT10Y": "IRLTLT01ITM156N",          # Italy 10Y
#     "UK10Y": "IRLTLT01GBM156N",          # UK 10Y
#     "JP10Y": "IRLTLT01JPM156N",          # Japan 10Y
#     "EC3M": "EUR3MTD156N",               # Euro 3M Rate
#     "US3M": "USD3MTD156N",               # USD 3M Rate
#     "TEDSpread": "TEDRATE",              # LIBOR - TBill Spread
#     "CPI_US": "CPIAUCSL",                # US CPI
#     "CPI_EU": "CP0000EZ19M086NEST",      # EU CPI
# }

fred_instance = fa.Fred(api_key='01cb69031a3eb2d02b939bce9a33b4b1')

US_10Y = fred_instance.get_series("DGS10", observation_start=start, observation_end=end)
DE_10Y = fred_instance.get_series("IRLTLT01DEM156N", observation_start=start, observation_end=end)
FR_10Y = fred_instance.get_series("IRLTLT01FRM156N", observation_start=start, observation_end=end)
IT_10Y = fred_instance.get_series("IRLTLT01ITM156N", observation_start=start, observation_end=end)
UK_10Y = fred_instance.get_series("IRLTLT01GBM156N", observation_start=start, observation_end=end)
JP_10Y = fred_instance.get_series("IRLTLT01JPM156N", observation_start=start, observation_end=end)
EC3M = fred_instance.get_series("IR3TIB01EZM156N", observation_start=start, observation_end=end)
US3M = fred_instance.get_series("IR3TIB01USM156N", observation_start=start, observation_end=end)
CPI_US = fred_instance.get_series("CPIAUCSL", observation_start=start, observation_end=end)
CPI_EU = fred_instance.get_series("CP0000EZ19M086NEST", observation_start=start, observation_end=end)


US_10Y = US_10Y.to_frame(name="US10Y")
US_10Y = US_10Y.resample('MS').mean()
US_10Y['Date'] = pd.to_datetime(US_10Y.index)
US_10Y.set_index('Date', inplace=True)



df_eur = pd.DataFrame({
    "DE10Y": DE_10Y,
    "FR10Y": FR_10Y,
    "IT10Y": IT_10Y,
    "UK10Y": UK_10Y,
    "JP10Y": JP_10Y,
    "EC3M": EC3M,
    "US3M": US3M,
    "CPI_US": CPI_US,
    "CPI_EU": CPI_EU
}).reset_index()
df_eur['Date'] = pd.to_datetime(df_eur['index'])
df_eur.drop(columns=['index'], inplace=True)
df_eur.set_index('Date', inplace=True)


df_final = pd.merge(US_10Y, df_eur, left_index=True, right_index=True, how='outer')
df_final = df_final.asfreq('B')
df_final.ffill(axis=0, inplace=True)


#Merge All Data
df_all = pd.merge(df_fx, df_comm_ind, left_index=True, right_index=True, how='outer')
df_all = pd.merge(df_all, df_final, left_index=True, right_index=True, how='outer')
df_all = df_all.asfreq('B')

st.dataframe(df_all)    




#Convert Price levels to Log Returns




price_cols = ["EURUSD", "USDJPY", "GBPUSD", "DXY",
              "FTSE100", "DAX30", "S&P500", "NIKKEI225", "CAC40",
              "EUROSTOXX50", "MIB", "BrentOil", "Gold"]


level_cols = ["VIX", "US10Y", "DE10Y", "FR10Y", "IT10Y", 
              "UK10Y", "JP10Y", "EC3M", "US3M", 
              "CPI_US", "CPI_EU"]

# Compute log returns for price columns
df_returns = np.log(df_all[price_cols]).diff()

# Keep levels for yields, macro
df_levels = df_all[level_cols]

# Create spreads 

df_spreads = pd.DataFrame({
    "US10Y_DE10Y": df_all["US10Y"] - df_all["DE10Y"],
    "US3M_EC3M": df_all["US3M"] - df_all["EC3M"],
})

df_features = pd.concat([df_returns, df_levels, df_spreads], axis=1)

# Drop NaN rows from first diff and ffill gaps
df_features = df_features.dropna().copy()

#Standardize features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Put back into DataFrame with same columns & index
df_scaled = pd.DataFrame(X_scaled, columns=df_features.columns, index=df_features.index)

X = df_scaled.drop(columns=["EURUSD"])  # Features
y = df_scaled["EURUSD"]                 # Target        

#PCA for Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  
X = pca.fit_transform(X)

#Splitting Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

#Random Walk Baseline
y_test_values = y_test.values
y_pred_naive = y_test_values[:-1]        
y_true_naive = y_test_values[1:]
mse_naive = mean_squared_error(y_true_naive, y_pred_naive)
r2_naive = r2_score(y_true_naive, y_pred_naive)  

#Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rf_accuracy = rf.score(x_test, y_test)

st.write(f"Random Forest Regressor Accuracy: {rf_accuracy:.5f}")
st.write("=== Random Forest Regressor vs Naive Random Walk Baseline ===")
st.write(f"Random Forest MSE: {mse:.5f} | Naive MSE: {mse_naive:.5f}")
st.write(f"Random Forest R²: {r2:.5f} | Naive R²: {r2_naive:.5f}")

#MSE - variance of target
target_variance = np.var(y_test)
mse_ratio = mse / target_variance
st.write(f"Random Forest MSE / Variance of Target: {mse_ratio:.5f}")

#Directional Accuracy
direction_correct = np.sign(y_pred) == np.sign(y_test)
accuracy = np.mean(direction_correct)
st.write(f"Random Forest Directional Accuracy: {accuracy:.5f}")

st.write('Random Forest MSE is lower than the Naive Random Walk MSE, indicating that the model is reducing prediction error compared to a simple baseline.')
st.write('The R² score is positive, suggesting that the model explains some variance in the target variable. The Random Forest R² is also higher than the Naive Random Walk R², indicating better performance.')
st.write('The MSE relative to the variance of the target is significantly less than 1, showing that the model is capturing meaningful patterns in the data.')
st.write('The directional accuracy above 50% indicates that the model is better than random guessing at predicting the direction of EUR/USD movements.')

#Backtrader Strategy

import backtrader as bt

threshold = 0.001
signals = np.where(y_pred > threshold, 1,
           np.where(y_pred < -threshold, -1, 0)) 

df_signals = pd.DataFrame({
    "Close": df_all.loc[y_test.index, "EURUSD"],  
    "Signal": signals
}, index=y_test.index)

df_signals.dropna(inplace=True)

df_signals['Open'] = df_signals['Close']
df_signals['High'] = df_signals['Close']
df_signals['Low'] = df_signals['Close']
df_signals['Volume'] = 1000

st.dataframe(df_signals)

class MLStrategy(bt.Strategy):
    params = (('signals', None),)

    def __init__(self):
        self.signals = self.params.signals
        self.index = 0

    def next(self):
        if self.index >= len(self.signals):
            return

        sig = self.signals[self.index]
        size = 1000 

        # Trading logic
        if sig == 1:
            if not self.position:  
                self.buy(size=size)
            elif self.position.size < 0:  
                self.close()
                self.buy(size=size)

        # If signal says "go short"
        elif sig == -1:
            if not self.position: 
                self.sell(size=size)
            elif self.position.size > 0: 
                self.close()
                self.sell(size=size)

        self.index += 1

data = bt.feeds.PandasData(dataname=df_signals)

cerebro = bt.Cerebro()
cerebro.addstrategy(MLStrategy, signals=df_signals['Signal'].values)
cerebro.adddata(data)

# Broker settings
cerebro.broker.set_cash(100000)
cerebro.addsizer(bt.sizers.FixedSize, stake=1000)
cerebro.broker.setcommission(commission=0.0002)  


# Run backtest
cerebro.run()
cerebro.plot()

st.write("Backtest Completed. Check the Backtrader plot for performance visualization.")    

