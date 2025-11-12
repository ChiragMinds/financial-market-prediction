# src/train.py
import argparse
import os
from src.data import download_close_prices, prepare_series_arrays, ensure_dirs
from src.model import build_model
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="AAPL,GOOG,MSFT,AMZN",
                   help="Comma-separated tickers to train on (default: AAPL,GOOG,MSFT,AMZN)")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2023-12-31")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save_dir", type=str, default="models")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dirs()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print("Downloading price series...")
    df = download_close_prices(tickers, start=args.start, end=args.end)
    X_dict, y_dict, scalers, raw = prepare_series_arrays(df, window_size=args.window, steps_ahead=args.steps)

    models = {}
    histories = {}

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
    ]

    for ticker in tickers:
        if ticker not in X_dict:
            print(f"Skipping {ticker}: no data.")
            continue

        X = X_dict[ticker]
        y = y_dict[ticker]
        print(f"\nTraining model for {ticker} — X shape: {X.shape}, y shape: {y.shape}")

        model = build_model(window_size=args.window, steps_ahead=args.steps, lr=args.lr)
        history = model.fit(X, y,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            validation_split=0.2,
                            verbose=1,
                            callbacks=callbacks)

        # save model & history
        model_path = os.path.join(args.save_dir, f"{ticker}_model.h5")
        os.makedirs(args.save_dir, exist_ok=True)
        model.save(model_path)
        np.save(os.path.join(args.save_dir, f"{ticker}_history.npy"), np.array(history.history['loss'], dtype=object))
        models[ticker] = model
        histories[ticker] = history

        # quick loss plot
        plt.figure(figsize=(8,3))
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title(f"Loss Curve - {ticker}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        os.makedirs(args.results_dir, exist_ok=True)
        plt.savefig(os.path.join(args.results_dir, f"{ticker}_loss.png"), bbox_inches='tight')
        plt.close()

        print(f"Saved model for {ticker} at {model_path} and loss plot.")

    print("\nTraining finished. Models saved to:", args.save_dir)
    print("Loss plots saved to:", args.results_dir)

if __name__ == "__main__":
    main()
