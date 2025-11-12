# src/evaluate.py
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from src.data import download_close_prices, prepare_series_arrays
import math

def directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))

def load_models_and_predict(tickers, window=50, steps=5, start='2018-01-01', end='2023-12-31', model_dir='models', sample_count=100, results_dir='results'):
    df = download_close_prices(tickers, start=start, end=end)
    X_dict, y_dict, scalers, raw = prepare_series_arrays(df, window_size=window, steps_ahead=steps)

    metrics_summary = {}
    os.makedirs(results_dir, exist_ok=True)

    for ticker in tickers:
        if ticker not in X_dict:
            print(f"Skipping {ticker}: no data available in prepared arrays.")
            continue
        model_path = os.path.join(model_dir, f"{ticker}_model.h5")
        if not os.path.exists(model_path):
            print(f"Model not found for {ticker}: {model_path}")
            continue

        model = load_model(model_path, compile=False)  # compile False since custom loss may not be available here
        X_vis = X_dict[ticker][:sample_count]
        y_vis = y_dict[ticker][:sample_count]
        y_pred = model.predict(X_vis)

        mae_list, rmse_list, mape_list, da_list = [], [], [], []
        for i in range(len(y_vis)):
            y_t = y_vis[i]
            y_p = y_pred[i]
            mae_list.append(mean_absolute_error(y_t, y_p))
            rmse_list.append(math.sqrt(mean_squared_error(y_t, y_p)))
            mape_list.append(np.mean(np.abs((y_t - y_p) / (y_t + 1e-5))) * 100)
            da_list.append(directional_accuracy(y_t, y_p) * 100)

        metrics_summary[ticker] = {
            'MAE': np.mean(mae_list),
            'RMSE': np.mean(rmse_list),
            'MAPE (%)': np.mean(mape_list),
            'Directional Accuracy (%)': np.mean(da_list)
        }

        # Save sample forecast plots (first 3)
        for i in range(min(3, len(y_vis))):
            plt.figure(figsize=(8,3))
            plt.plot(range(steps), y_vis[i], label="Actual", marker='o')
            plt.plot(range(steps), y_pred[i], label="Predicted", marker='o')
            plt.title(f"{ticker} Forecast Sample {i+1}")
            plt.xlabel("Step ahead")
            plt.ylabel("Scaled Price")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{ticker}_forecast_{i+1}.png"), bbox_inches='tight')
            plt.close()

    # Comparative bar chart
    tickers_evaluated = list(metrics_summary.keys())
    mae_scores = [metrics_summary[t]['MAE'] for t in tickers_evaluated]
    rmse_scores = [metrics_summary[t]['RMSE'] for t in tickers_evaluated]
    mape_scores = [metrics_summary[t]['MAPE (%)'] for t in tickers_evaluated]
    da_scores = [metrics_summary[t]['Directional Accuracy (%)'] for t in tickers_evaluated]

    metrics = ['MAE', 'RMSE', 'MAPE (%)', 'Directional Accuracy (%)']
    data = [mae_scores, rmse_scores, mape_scores, da_scores]

    plt.figure(figsize=(12,6))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 4, i+1)
        plt.bar(tickers_evaluated, data[i], color='skyblue')
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()

    plt.suptitle("Comparative Analysis of Stock Prediction Metrics", fontsize=14, y=1.05)
    plt.savefig(os.path.join(results_dir, "comparative_metrics.png"), bbox_inches='tight')
    plt.close()

    # print metrics summary
    for t in tickers_evaluated:
        m = metrics_summary[t]
        print(f"\nEvaluation for {t}:")
        print(f"MAE:  {m['MAE']:.4f}")
        print(f"RMSE: {m['RMSE']:.4f}")
        print(f"MAPE: {m['MAPE (%)']:.2f}%")
        print(f"Directional Accuracy: {m['Directional Accuracy (%)']:.2f}%")

    return metrics_summary

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="AAPL,GOOG,MSFT,AMZN")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2023-12-31")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--results_dir", type=str, default="results")
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    load_models_and_predict(tickers, window=args.window, steps=args.steps, start=args.start, end=args.end,
                            model_dir=args.model_dir, sample_count=100, results_dir=args.results_dir)
