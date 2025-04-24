import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import joblib
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import RobustScaler
import os

# Load models and scaler
custom_objects = {"mse": MeanSquaredError()}
lstm_model = load_model("models/lstm_model.h5", custom_objects=custom_objects)
linear_regression = joblib.load("models/linear_regression_model.pkl")
random_forest = joblib.load("models/optimized_random_forest_model.pkl")
xgboost = joblib.load("models/optimized_xgboost_model.pkl")
lightgbm = joblib.load("models/optimized_lightgbm_model.pkl")
gradient_boost = joblib.load("models/gradient_boosting_model.pkl")
svr = joblib.load("models/svr_model.pkl")
scaler_X = joblib.load("models/scaler_X.pkl")

# Analyzer
analyzer = SentimentIntensityAnalyzer()

# Helper functions
def analyze_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    return score['compound'], score['neg'], score['neu'], score['pos']

def fetch_news_sentiment(company, date):
    NEWS_API_KEY = '180ccde2a0d942048f65a588d9d03470'
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': company,
        'from': date,
        'to': date,
        'language': 'en',
        'pageSize': 5,
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    sentiments = [analyze_sentiment(article['content']) for article in response.json().get('articles', []) if article.get('content')]
    if sentiments:
        sentiments_df = pd.DataFrame(sentiments, columns=['sentiment_polarity', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos'])
        return sentiments_df.mean()
    return pd.Series([0, 0, 0, 0], index=['sentiment_polarity', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos'])

def get_technical_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_line'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
    df['RSI'] = df['Close'].diff().rolling(window=14).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    return df.dropna()

def make_predictions(features):
    predictions = {
        "Linear Regression": linear_regression.predict(features)[0],
        "Random Forest": random_forest.predict(features)[0],
        "XGBoost": xgboost.predict(features)[0],
        "LightGBM": lightgbm.predict(features)[0],
        "SVR": svr.predict(features)[0],
        "Gradient Boosting": gradient_boost.predict(features)[0],
    }
    lstm_input = np.array(features).reshape(1, 1, -1)
    predictions["LSTM"] = lstm_model.predict(lstm_input)[0][0]
    return predictions

# Target company and timeframe
ticker = "META"
company_name = "Facebook"
end_date = datetime.today()
start_date = end_date - timedelta(days=30)
df = yf.Ticker(ticker).history(start=start_date, end=end_date)
df = get_technical_indicators(df)

# Predict for each day
results = []
for i in range(len(df)):
    row = df.iloc[i]
    date = row.name.strftime('%Y-%m-%d')
    sentiment = fetch_news_sentiment(company_name, date)
    feature_row = pd.DataFrame([{
        "RSI": row["RSI"],
        "MACD_line": row["MACD_line"],
        "MACD_signal": row["MACD_signal"],
        "MACD_hist": row["MACD_hist"],
        "MA20": row["MA20"],
        "sentiment_polarity": sentiment["sentiment_polarity"],
        "sentiment_neg": sentiment["sentiment_neg"],
        "sentiment_neu": sentiment["sentiment_neu"],
        "sentiment_pos": sentiment["sentiment_pos"]
    }])
    scaled_input = scaler_X.transform(feature_row)
    preds = make_predictions(scaled_input)
    preds["Actual Price"] = row["Close"]
    preds["Date"] = date
    results.append(preds)

# Convert results list to DataFrame
predictions_df = pd.DataFrame(results)

# Show in Streamlit
st.subheader("  30-Day Prediction vs Actual Comparison")
st.dataframe(predictions_df.style.format("{:.2f}"))