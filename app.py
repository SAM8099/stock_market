import streamlit as st 
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipelines.predict_pipeline import PredictPipeline, CustomData
from src.pipelines.train_pipeline import train
from src.exception import CustomException
import yfinance as yf
import pandas as pd
import os
import ta
from datetime import datetime, timedelta
from src.utils import load_object
import sys

st.title("📈 Mini Algo-Trading App")
st.markdown("""
This app fetches historical stock data, applies an RSI + MA strategy, logs trades,
and uses a simple ML model to predict stock movement.
""")

# Sidebar options
st.sidebar.header("Configuration")
stock_symbol = st.sidebar.selectbox("Select Stock", ['RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS'])
def get_today_stock_features(stock_symbol: str):
    try:
        # Set today's date
        target_date = datetime.today().strftime('%Y-%m-%d')

        # Get data for last 90 days to ensure indicators are calculated

        df = yf.Ticker(stock_symbol).history(period = '90d', interval='1d')

        if df.empty:
            raise ValueError("No data fetched for given symbol and date range")

        # Compute indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        df.dropna(inplace=True)  # Drop initial rows with NaNs

        df.index = df.index.strftime('%Y-%m-%d')  # Convert index for comparison

        if target_date not in df.index:
            # If market hasn't opened today, fallback to latest available row
            print(f"Today's data not available yet. Using last available date: {df.index[-1]}")
            target_date = df.index[-1]

        row = df.loc[target_date]
        row_df = pd.DataFrame([row])  # Single-row DataFrame for prediction
        row_df.drop(columns=['Date', 'Dividends', 'Stock Splits', 'Buy', 'Sell'], inplace=True, errors='ignore')  # Add date column
        return row_df # Include more features as needed

    except Exception as e:
            raise CustomException(e, sys)
            raise CustomException(e, sys)
model = PredictPipeline(stock_symbol=stock_symbol)
if st.sidebar.button("Predict"):
    try:
        data = get_today_stock_features(stock_symbol)
        if data.empty:
            st.error("No data available for today's date. Please try again later.")
            st.stop()
        prediction = model.predict(data)
        st.write(f"Prediction for {stock_symbol}: {'Buy' if prediction[0] == 1 else 'Sell'}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")



