# src/data.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


def fetch_stock_data():
    """
    Downloads the 4 default tickers used in the notebook:
    AAPL, GOOG, MSFT, AMZN
    Returns a cleaned DataFrame of Close prices.
    """
    tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
    print(f"Downloading data for: {', '.join(tickers)}")

    data = yf.download(tickers, start='2018-01-01', end='2023-12-31')['Close'].dropna()
    print(f"Data downloaded: {data.shape[0]} rows × {data.shape[1]} columns")

    # Optional: visualize
    data.plot(figsize=(12, 5), title='Historical Stock Prices')
    plt.grid(True)
    plt.show()

    return data


def prepare_data(data, window_size=50, steps_ahead=5):
    """
    Converts the Close price DataFrame into training-ready arrays per ticker.
    """
    X_dict, y_dict, scalers = {}, {}, {}

    for ticker in data.columns:
        series = data[ticker].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series).flatten()

        X, y = [], []
        for i in range(len(scaled) - window_size - steps_ahead + 1):
            X.append(scaled[i:i + window_size])
            y.append(scaled[i + window_size:i + window_size + steps_ahead])

        X = np.expand_dims(np.array(X), axis=-1)
        y = np.array(y)

        X_dict[ticker] = X
        y_dict[ticker] = y
        scalers[ticker] = scaler

        print(f"{ticker}: X={X.shape}, y={y.shape}")

    return X_dict, y_dict, scalers


def ensure_dirs():
    """
    Creates folders for models and results if they don’t exist.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
