import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import pandas as pd
import nltk
from newspaper import Article
from GoogleNews import GoogleNews
from textblob import TextBlob
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import RobustScaler


# Define loss function explicitly
custom_objects = {"mse": MeanSquaredError()}

# Load the LSTM model
lstm_model_loaded = load_model("models/lstm_model.h5", custom_objects=custom_objects)

# Load each machine learning model individually
linear_regression_model_loaded = joblib.load("models/linear_regression_model.pkl")
optimized_random_forest_model_loaded = joblib.load("models/optimized_random_forest_model.pkl")
optimized_xgboost_model_loaded = joblib.load("models/optimized_xgboost_model.pkl")
optimized_lightgbm_model_loaded = joblib.load("models/optimized_lightgbm_model.pkl")
gradient_boosting_model_loaded = joblib.load("models/gradient_boosting_model.pkl")
svr_model_loaded = joblib.load("models/svr_model.pkl")

# Load the Prophet model
prophet_model_loaded = joblib.load("models/prophet_model.pkl")

#     Load Scaler (Assuming RobustScaler was used in training)
scaler = RobustScaler()

#     Fetch Real-Time Stock Data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="21d")  # Last 
    return df

#     Fetch Google News Headlines & Perform Sentiment Analysis
def get_news_sentiment(ticker):
    googlenews = GoogleNews(lang='en', period='7d')
    googlenews.search(ticker)
    news_list = googlenews.result()
    
    sentiments = {"polarity": [], "neg": [], "neu": [], "pos": []}
    
    for news in news_list[:10]:  # Limit to 10 articles
        try:
            article = Article(news["link"])
            article.download()
            article.parse()
            article.nlp()
            text = article.text
            sentiment = TextBlob(text).sentiment
            sentiments["polarity"].append(sentiment.polarity)
            sentiments["neg"].append(1 if sentiment.polarity < 0 else 0)
            sentiments["neu"].append(1 if sentiment.polarity == 0 else 0)
            sentiments["pos"].append(1 if sentiment.polarity > 0 else 0)
        except:
            continue
    
    return pd.DataFrame(sentiments).mean()

#     Predict Stock Price using Loaded Models
def predict_stock_price(features):
    # Replace NaN values with column median or zero
    features = np.nan_to_num(features, nan=0.0)  # Replace NaNs with 0.0

    predictions = {
        "Linear Regression": linear_regression_model_loaded.predict(features)[0],
        "Random Forest": optimized_random_forest_model_loaded.predict(features)[0],
        "XGBoost": optimized_xgboost_model_loaded.predict(features)[0],
        "LightGBM": optimized_lightgbm_model_loaded.predict(features)[0],
        "SVR": svr_model_loaded.predict(features)[0],
        "Gradient Boosting": gradient_boosting_model_loaded.predict(features)[0],
        "Prophet": prophet_model_loaded.predict(features)[0]
    }
    
    #     LSTM Prediction
    lstm_features = np.array(features).reshape(1, 1, -1)
    lstm_pred = lstm_model_loaded.predict(lstm_features)[0][0]
    predictions["LSTM"] = lstm_pred
    
    return predictions

#     Streamlit Web App
st.title("    Real-Time Stock Price Prediction")
st.sidebar.header("Select FAANG Company")
ticker = st.sidebar.selectbox("Choose a stock", ["AAPL", "GOOGL", "META", "NFLX", "AMZN"])

if st.sidebar.button("Predict Stock Price"):
    with st.spinner("Fetching data & making predictions..."):
        stock_data = get_stock_data(ticker)
        print(stock_data)
        stock_data.dropna(inplace=True)  # Remove any missing values in stock data
        
        news_sentiment = get_news_sentiment(ticker)
        print(news_sentiment)
        if news_sentiment.isnull().values.any():
             news_sentiment.fillna(0, inplace=True)  # Fill missing sentiment values with 0

        #     Prepare Features for Prediction
        latest_price = stock_data["Close"].iloc[-1]
        rsi = stock_data["Close"].diff().rolling(window=14).mean().iloc[-1]
        macd = stock_data["Close"].ewm(span=12).mean().iloc[-1] - stock_data["Close"].ewm(span=26).mean().iloc[-1]
        ma20 = stock_data["Close"].rolling(window=20).mean().iloc[-1]

        features = pd.DataFrame([[rsi, macd, ma20, news_sentiment["polarity"], news_sentiment["neg"],
                         news_sentiment["neu"], news_sentiment["pos"]]], 
                        columns=["RSI", "MACD", "MA20", "sentiment_polarity", 
                                 "sentiment_neg", "sentiment_neu", "sentiment_pos"]
                        ).fillna(0)
        
        print(features)
        print(features.isnull().sum())  # Check missing values

        # Scale Features
        features_scaled = scaler.fit_transform(features)
        
        # Make Predictions
        predictions = predict_stock_price(features_scaled)

        # Display Results
        st.subheader(f"  Predicted Stock Prices for {ticker}")
        for model, pred_price in predictions.items():
            st.write(f"  {model}: **${pred_price:.2f}**")

        # Fetch Real Price for Comparison
        actual_price = stock_data["Close"].iloc[-1]
        st.subheader(f"    Actual Closing Price: **${actual_price:.2f}**")
