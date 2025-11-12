# src/evaluate.py
"""
Evaluation script: loads saved models and computes metrics & example forecasts.
Usage:
python src/evaluate.py --tickers "AAPL,GOOG,MSFT,AMZN" --start 2018-01-01 --end 2023-12-31 --window 50 --steps 5
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from src.data import download_close_prices, prepare_series_arrays, fetch_stock_data
from src.data import prepare_series_arrays  # ensure correct import
import math

def directional_accuracy(y_true, y_pred):
    """
    Compute directional accuracy (%) comparing successive steps.
    y_true, y_pred: arrays shape (steps,)
    """
    if len(y_true) <= 1 or len(y_pred) <= 1:
        return 0.0
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100.0

def evaluate_models(tickers, window=50, steps=5, start='2018-01-01', end='2023-12-31',
                    model_dir='models', results_dir='results', sample_count=100):
    os.makedirs(results_dir, exist_ok=True)

    # Fetch series and prepare arrays
    df = fetch_stock_data(tickers=tickers, start=start, end=end, plot=False)
    X_dict, y_dict, scalers, raw = prepare_series_arrays(df, window_size=window, steps_ahead=steps)

    summary = {}

    for ticker in tickers:
        if ticker not in X_dict:
            print(f"Skipping {ticker}: no data windows prepared.")
            continue

        model_path = os.path.join(model_dir, f"{ticker}_model.h5")
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Skipping {ticker}.")
            continue

        model = load_model(model_path, compile=False)

        X = X_dict[ticker]
        y = y_dict[ticker]

        # limit samples for quick evaluation
        n = min(sample_count, X.shape[0])
        X_s = X[:n]
        y_s = y[:n]

        preds = model.predict(X_s)

        mae_vals, rmse_vals, mape_vals, da_vals = [], [], [], []

        for i in range(len(y_s)):
            true = y_s[i]
            pred = preds[i]
            mae = mean_absolute_error(true, pred)
            rmse = math.sqrt(mean_squared_error(true, pred))
            mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100.0
            da = directional_accuracy(true, pred)

            mae_vals.append(mae)
            rmse_vals.append(rmse)
            mape_vals.append(mape)
            da_vals.append(da)

        summary[ticker] = {
            'MAE': float(np.mean(mae_vals)),
            'RMSE': float(np.mean(rmse_vals)),
            'MAPE (%)': float(np.mean(mape_vals)),
            'Directional Accuracy (%)': float(np.mean(da_vals))
        }

        # Plot sample forecasts (first 3)
        for i in range(min(3, len(y_s))):
            plt.figure(figsize=(6,3))
            plt.plot(range(steps), y_s[i], marker='o', label='Actual')
            plt.plot(range(steps), preds[i], marker='o', label='Predicted')
            plt.title(f"{ticker} Forecast sample {i+1}")
            plt.xlabel("Steps ahead")
            plt.ylabel("Scaled price")
            plt.legend()
            fname = os.path.join(results_dir, f"{ticker}_forecast_{i+1}.png")
            plt.savefig(fname, bbox_inches='tight')
            plt.close()

    # comparative bar chart
    tickers_evaluated = list(summary.keys())
    if len(tickers_evaluated) > 0:
        metrics = ['MAE', 'RMSE', 'MAPE (%)', 'Directional Accuracy (%)']
        plt.figure(figsize=(14, 4))
        for i, metric in enumerate(metrics):
            plt.subplot(1, 4, i+1)
            values = [summary[t][metric] for t in tickers_evaluated]
            plt.bar(tickers_evaluated, values, color='skyblue')
            plt.title(metric)
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'comparative_metrics.png'), bbox_inches='tight')
        plt.close()

    # print summary
    for t in summary:
        print(f"\n{t}:")
        for k, v in summary[t].items():
            print(f"  {k}: {v:.4f}")

    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="AAPL,GOOG,MSFT,AMZN")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2023-12-31")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--sample_count", type=int, default=100)
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    evaluate_models(tickers, window=args.window, steps=args.steps,
                    start=args.start, end=args.end, model_dir=args.model_dir,
                    results_dir=args.results_dir, sample_count=args.sample_count)
