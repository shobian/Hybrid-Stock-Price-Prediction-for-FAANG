import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta
import os

#     Load Models
custom_objects = {"mse": MeanSquaredError()}
lstm_model = load_model("models/lstm_model.h5", custom_objects=custom_objects)
linear_regression = joblib.load("models/linear_regression_model.pkl")
random_forest = joblib.load("models/optimized_random_forest_model.pkl")
xgboost = joblib.load("models/optimized_xgboost_model.pkl")
lightgbm = joblib.load("models/optimized_lightgbm_model.pkl")
gradient_boost = joblib.load("models/gradient_boosting_model.pkl")
svr = joblib.load("models/svr_model.pkl")

scaler_y = joblib.load("models/scaler_y.pkl")
scaler_X = joblib.load("models/scaler_X.pkl")

#  News Sentiment Fetching from NewsAPI
NEWS_API_KEY = '180ccde2a0d942048f65a588d9d03470'

def fetch_faang_news_sentiment():
    """
    Fetch the last 30 days of FAANG-related news.
    Here we just average all 30 days together in one shot.
    If you prefer daily-level sentiment, you’d need a more sophisticated approach
    that queries news by date and aggregates day by day.
    """
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'Apple',
        'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        'to': datetime.now().strftime('%Y-%m-%d'),
        'language': 'en',
        'pageSize': 20,
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        print(articles)
        return pd.DataFrame(articles)
    else:
        print("no article found in the last 30 days")
        return pd.DataFrame()

#   Sentiment Analyzer
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(str(text))
    return score['compound'], score['neg'], score['neu'], score['pos']

def prepare_sentiment_features(df):
    """
    Returns a single average set of sentiment features. For daily-level
    sentiment, you would need to modify to handle each day separately.
    """
    if df.empty or 'content' not in df:
        return pd.Series([0, 0, 0, 0], 
                         index=['sentiment_polarity', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos'])
    df[['sentiment_polarity', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos']] = (
        df['content'].fillna("").apply(lambda x: pd.Series(analyze_sentiment(x)))
    )
    print("sentiment_polarity\n", df)
    return df[['sentiment_polarity','sentiment_neg','sentiment_neu','sentiment_pos']].mean()

#     Technical Indicator Calculations for the entire DataFrame
def compute_technical_indicators(df):
    """
    Compute the indicators (MACD, RSI, etc.) for every row, then drop NaN.
    """
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_line'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
    # Simple RSI example (you could replace this with a more accurate RSI calc if needed)
    df['RSI'] = df['Close'].diff().rolling(window=14).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    df.dropna(inplace=True)
    print(df)
    return df

#   Predict Function
def make_predictions(features):
    predictions = {}
    feat_2d = features.reshape(1, -1)

    predictions["Linear Regression"] = scaler_y.inverse_transform([[linear_regression.predict(feat_2d)[0]]])[0][0]
    predictions["Random Forest"] = scaler_y.inverse_transform([[random_forest.predict(feat_2d)[0]]])[0][0]
    predictions["XGBoost"] = scaler_y.inverse_transform([[xgboost.predict(feat_2d)[0]]])[0][0]
    predictions["LightGBM"] = scaler_y.inverse_transform([[lightgbm.predict(feat_2d)[0]]])[0][0]
    predictions["SVR"] = scaler_y.inverse_transform([[svr.predict(feat_2d)[0]]])[0][0]
    predictions["Gradient Boosting"] = scaler_y.inverse_transform([[gradient_boost.predict(feat_2d)[0]]])[0][0]

    lstm_input = features.reshape(1, 1, -1)
    lstm_raw = lstm_model.predict(lstm_input)[0][0]
    predictions["LSTM"] = scaler_y.inverse_transform([[lstm_raw]])[0][0]

    return predictions


#  Streamlit UI
st.title("  FAANG Stock Price Predictor")
st.sidebar.header("  Choose a FAANG Company")
ticker = st.sidebar.selectbox("Select Stock Ticker", ["AAPL", "GOOGL", "META", "NFLX", "AMZN"])

if st.sidebar.button("    Predict Last 30 Days"):
    with st.spinner("Fetching data for last 60 days, computing indicators, and predicting..."):
        # 1) Fetch 60 days of data
        df_raw = yf.Ticker(ticker).history(period="60d")
        if df_raw.empty:
            st.error("No data retrieved from yfinance.")
            st.stop()
        
        # 2) Compute technicals for entire DataFrame
        df = compute_technical_indicators(df_raw.copy())

        # If there's insufficient data after dropna, handle gracefully
        if len(df) < 30:
            st.error("Not enough data to predict 30 days. Please try again later.")
            st.stop()

        # 3) Prepare or fetch average sentiment for all days (example: single average for entire 30d)
        # For daily sentiment, you'd need a more advanced approach
        sentiment_data = fetch_faang_news_sentiment()
        sentiment_avg = prepare_sentiment_features(sentiment_data)

        # Build the results list
        predictions_list = []

        # We'll just do the last 30 days from the available df
        last_30 = df.iloc[-30:].copy()
        # For each row in last_30, build feature vector
        for date_idx, row in last_30.iterrows():
            # We create the feature row that matches the training pipeline
            # same column order that your scaler expects
            feats = [
                row["RSI"],
                row["MACD_line"],
                row["MACD_signal"],
                row["MACD_hist"],
                row["MA20"],
                sentiment_avg["sentiment_polarity"],
                sentiment_avg["sentiment_neg"],
                sentiment_avg["sentiment_neu"],
                sentiment_avg["sentiment_pos"]
            ]
            feats = np.array(feats, dtype=float)
            scaled_feats = scaler_X.transform([feats])[0]  # shape -> (9,)

            model_preds = make_predictions(scaled_feats)
            
            # Actual close or next-day close?
            # If you want to compare with the same day, do row['Close'] 
            # If you want next day, you might look up the next row from df (and handle edge cases)
            actual_price = row["Close"]
            
            # Collect everything in a dictionary
            day_result = {
                "Date": date_idx.date(),
                "Actual_Close": actual_price
            }
            # Merge model predictions
            for model_name, pred_price in model_preds.items():
                day_result[model_name] = pred_price
            predictions_list.append(day_result)

        # Convert to DataFrame
        results_df = pd.DataFrame(predictions_list)
        st.write("### 30-Day Predictions vs. Actual")
        st.dataframe(results_df.reset_index(drop=True))

        st.write("You can scroll right to see all models’ predictions.")
        st.success("Done!")
