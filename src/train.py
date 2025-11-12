# src/train.py
"""
Training script.

Usage example:
python src/train.py --tickers "AAPL,GOOG,MSFT,AMZN" --start "2018-01-01" --end "2023-12-31" --epochs 50 --batch_size 256
"""

import argparse
import os
import numpy as np
from src.data import fetch_stock_data, prepare_series_arrays, ensure_dirs
from src.model import build_model
from src.utils import save_model, save_history, plot_loss, save_scaler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="AAPL,GOOG,MSFT,AMZN",
                   help="Comma-separated tickers (default: AAPL,GOOG,MSFT,AMZN)")
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

def train_for_ticker(ticker, X, y, args):
    """
    Train model for a single ticker using provided arrays.
    """
    print(f"Training {ticker} - X shape: {X.shape}, y shape: {y.shape}")
    model = build_model(window_size=args.window, steps_ahead=args.steps, lr=args.lr)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(X, y,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=1)

    # save model, history and loss plot
    model_path = os.path.join(args.save_dir, f"{ticker}_model.h5")
    save_model(model, model_path)
    hist_path = os.path.join(args.save_dir, f"{ticker}_history.npy")
    save_history(history.history, hist_path)
    loss_plot_path = os.path.join(args.results_dir, f"{ticker}_loss.png")
    plot_loss(history, loss_plot_path)

    return model, history

def main():
    args = parse_args()
    ensure_dirs()

    # download data for all tickers together
    tickers_list = [t.strip().upper() for t in args.tickers.split(",")]
    df = fetch_stock_data(tickers=tickers_list, start=args.start, end=args.end, plot=False)

    X_dict, y_dict, scalers, raw = prepare_series_arrays(df, window_size=args.window, steps_ahead=args.steps)

    for ticker in tickers_list:
        if ticker not in X_dict:
            print(f"Skipping {ticker}: no training windows (check data length)")
            continue
        X = X_dict[ticker]
        y = y_dict[ticker]
        model, history = train_for_ticker(ticker, X, y, args)
        # Save scaler for this ticker for future inverse-transforms
        scaler_path = os.path.join(args.save_dir, f"{ticker}_scaler.save")
        save_scaler(scalers[ticker], scaler_path)

    print("All training complete. Models saved in:", args.save_dir)
    print("Loss plots saved in:", args.results_dir)

if __name__ == "__main__":
    main()
