# src/data.py
"""
Data module: downloads close prices for the four tickers used in the notebook
and converts series into sliding-window arrays for model training.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


def fetch_stock_data(tickers=None, start='2018-01-01', end='2023-12-31', plot=True):
    """
    Downloads daily Close prices for the provided tickers using yfinance.
    Default tickers: ['AAPL', 'GOOG', 'MSFT', 'AMZN']

    Returns
    -------
    pd.DataFrame: DataFrame of Close prices with columns as tickers.
    """
    if tickers is None:
        tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

    if isinstance(tickers, str):
        # Allow comma-separated string too
        tickers = [t.strip().upper() for t in tickers.split(",")]

    print(f" Downloading data for: {', '.join(tickers)} (from {start} to {end})")
    df = yf.download(tickers, start=start, end=end)['Close']

    if df is None or df.empty:
        raise ValueError("No data returned from yfinance. Check tickers and date range.")

    # If single ticker, yfinance returns a Series: convert it to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])

    # Drop rows where all columns are NaN (market holidays)
    df = df.dropna(how='all')

    print(f" Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    if plot:
        try:
            df.plot(figsize=(12, 5), title='Historical Stock Prices')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print("Plotting failed:", e)

    return df


def prepare_series_arrays(df, window_size=50, steps_ahead=5, min_windows=1):
    """
    Convert each ticker series into sliding window arrays for supervised training.

    Returns:
      X_dict: dict[ticker] -> np.ndarray (n_samples, window_size, 1)
      y_dict: dict[ticker] -> np.ndarray (n_samples, steps_ahead)
      scalers: dict[ticker] -> MinMaxScaler fitted on that series
      raw: dict[ticker] -> original 1D numpy array (unscaled)
    """
    X_dict, y_dict, scalers, raw = {}, {}, {}, {}

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame with tickers as columns.")

    for ticker in df.columns:
        series = df[ticker].dropna().values.reshape(-1, 1).astype(float)
        if len(series) < (window_size + steps_ahead):
            print(f"  Skipping {ticker}: series too short ({len(series)} points).")
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series).flatten()

        X_list, y_list = [], []
        max_i = len(scaled) - window_size - steps_ahead + 1
        for i in range(max_i):
            X_list.append(scaled[i:i + window_size])
            y_list.append(scaled[i + window_size: i + window_size + steps_ahead])

        if len(X_list) < min_windows:
            print(f"  No sufficient windows for {ticker}. Created {len(X_list)} windows. Skipping.")
            continue

        X = np.expand_dims(np.array(X_list), axis=-1)  # (N, window_size, 1)
        y = np.array(y_list)                            # (N, steps_ahead)
        raw_series = series.flatten()

        X_dict[ticker] = X
        y_dict[ticker] = y
        scalers[ticker] = scaler
        raw[ticker] = raw_series

        print(f" {ticker}: X shape {X.shape}, y shape {y.shape}")

    if len(X_dict) == 0:
        raise RuntimeError("No tickers produced training windows. Check data and parameters.")

    return X_dict, y_dict, scalers, raw


def ensure_dirs():
    """Create models and results directories if they do not exist."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
