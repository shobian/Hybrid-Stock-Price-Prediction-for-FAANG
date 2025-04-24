#############################
# Apple Stock Prediction Notebook
#############################

# If running in a script, you can save this as "Apple_Prediction.ipynb"
# or a standard .py, depending on your environment.

#############################
# 1) Import Libraries
#############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For technical indicator computations
# (We can write custom RSI, MACD, or use a library like pandas_ta)
# Here, we will code them manually for clarity.

#############################
# 2) Helper Functions: RSI, MACD, MA
#############################

def compute_rsi(series, period=14):
    """
    Computes the RSI (Relative Strength Index) for a given
    price series. Common default is 14-day RSI.
    """
    delta = series.diff()
    # Gains (positive deltas) & losses (negative deltas)
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Exponential moving average of gains/losses:
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    # If using exponential weighting, we could do gains.ewm(...)
    # For simplicity, using simple rolling average approach here

    rs = avg_gain / (avg_loss + 1e-9)  # add small epsilon to avoid /0
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Computes MACD (Moving Average Convergence Divergence).
    Returns: macd_line, signal_line, macd_hist
    """
    # Exponential Moving Average (EMA) or simple if needed
    # We'll do simple for demonstration, but typically MACD uses EMA
    # For a quick approach, let's do standard EMAs:
    
    # A quick EMA function
    def ema(a, span):
        return a.ewm(span=span, adjust=False).mean()
    
    ema_fast = ema(series, fastperiod)
    ema_slow = ema(series, slowperiod)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_ma(series, window=20):
    """ Simple Moving Average for a given window. """
    return series.rolling(window=window).mean()

#############################
# 3) Load and Clean the Data
#############################

# We will assume the CSVs are in the same directory or provide correct paths.

# A) Load Apple Price Data
# NOTE: The user described a complicated header in AAPL.csv. We'll do:
# We'll skip the first row which might be "Price,Close,High,Low" repeated, etc.
# Adjust skiprows if needed, or read everything and rename columns.

df_price_raw = pd.read_csv("AAPL.csv", header=None)  # read everything first

# Inspect first few rows to see how to handle them. (In practice you'd do df_price_raw.head())
# The user snippet suggests row0 = [Price,Close,High,Low,...], row1 = [Ticker, AAPL, ...], row2 = actual data
# We'll do a trick: skip first 2 lines, and then name columns ourselves.

df_price = pd.read_csv("AAPL.csv", skiprows=2, header=None)
# The columns might be something like: Date,Open,High,Low,Close,Volume
# Let's rename them carefully:

df_price.columns = ["Date","Close","High","Low","Open","Something_unused","Something_unused2","Volume","Unused3","Unused4"]
# This depends on the actual structure. The user’s snippet wasn't fully standard.
# If we only want: Date, Open, High, Low, Close, Volume, let's pick them:

df_price = df_price[["Date","Open","High","Low","Close","Volume"]]
df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce")
df_price.dropna(subset=["Date"], inplace=True)
df_price.sort_values("Date", inplace=True)
df_price.reset_index(drop=True, inplace=True)

# B) Load Apple News Data
df_news = pd.read_csv("apple_news_data.csv")
# We'll assume there's a 'Date' column plus 'sentiment_polarity','sentiment_neg','sentiment_neu','sentiment_pos', etc.

df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce")
df_news.dropna(subset=["Date"], inplace=True)
df_news.sort_values("Date", inplace=True)
df_news.reset_index(drop=True, inplace=True)

#############################
# 4) Merge Price and News Data
#############################

# We'll merge on "Date". If the news is daily, we can do a left merge or an outer merge.
# Typically, price data is daily, and news might be multiple per day, but let's assume daily aggregated sentiments.

df_merged = pd.merge(df_price, df_news, on="Date", how="inner")  # or "left"

#############################
# 5) Create Technical Indicators
#############################

df_merged["RSI"] = compute_rsi(df_merged["Close"], period=14)
macd_line, signal_line, macd_hist = compute_macd(df_merged["Close"])
df_merged["MACD_line"] = macd_line
df_merged["MACD_signal"] = signal_line
df_merged["MACD_hist"] = macd_hist

# Let's create a 20-day MA
df_merged["MA20"] = compute_ma(df_merged["Close"], window=20)

# We may have some NaNs at the start due to rolling windows, drop them:
df_merged.dropna(inplace=True)
df_merged.reset_index(drop=True, inplace=True)

#############################
# 6) Define Our Target (Next-Day Direction)
#############################

# We'll create a simple binary classification: 1 if next day's Close > today's Close, else 0
# SHIFT the "Close" by -1 to get tomorrow's close
df_merged["Close_next"] = df_merged["Close"].shift(-1)
df_merged.dropna(subset=["Close_next"], inplace=True)  # last row becomes NaN
df_merged["Target_UpDown"] = (df_merged["Close_next"] > df_merged["Close"]).astype(int)

#############################
# 7) Features and Labels
#############################

# Let's define our feature columns to use: RSI, MACD_line, MACD_signal, MACD_hist, MA20
# plus the sentiment columns: sentiment_polarity, sentiment_neg, sentiment_neu, sentiment_pos
feature_cols = [
    "RSI",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    "MA20",
    "sentiment_polarity",
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_pos"
]

X = df_merged[feature_cols].copy()
y = df_merged["Target_UpDown"].copy()

#############################
# 8) Train-Test Split
#############################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)
# shuffle=False to respect time order (some prefer rolling or time-series split)
# For a simple demonstration, we do a hold-out from the end.

#############################
# 9) Build a Random Forest Classifier
#############################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

#############################
# 10) Evaluate the Model
#############################

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Accuracy on test set:", acc)
print("Classification Report:\n", classification_report(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# If desired, you can compute predicted probabilities and define
# your own threshold-based signals or explore more advanced metrics.

#############################
# 11) Quick Visualization (Optional)
#############################

plt.figure(figsize=(10,4))
plt.plot(df_merged["Date"], df_merged["Close"], label="Close Price", color="blue")
plt.title("AAPL Close Price Over Time (Merged Data)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

#############################
# Done
#############################