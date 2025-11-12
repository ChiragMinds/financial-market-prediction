# 📈 LSTM Networks for Financial Market Prediction

**Project summary**  
This repository contains a modular conversion of an experimental notebook into a reproducible project that trains sequence models for short-term financial market prediction. The code downloads historical price data for four tickers (AAPL, GOOG, MSFT, AMZN) using `yfinance`, prepares sliding-window sequences, trains a deep neural network (Conv1D + BiLSTM + attention) per ticker, and evaluates performance using multiple metrics (MAE, RMSE, MAPE, directional accuracy).

This repository is intended for **demonstration and academic use**. The code, models, and outputs are organized for clarity and reproducibility.

---

## 🔧 Features and components
- Internal data fetching with `yfinance` (no local dataset required).
- Data preprocessing, sliding-window sequence creation per ticker.
- Modular model builder (Conv1D + Bidirectional LSTM + Attention).
- Training script with callbacks, per-ticker model saving, and loss plots.
- Evaluation script that produces numeric metrics and forecast visualizations.
- Structured output: trained models saved in `models/`, plots in `results/`.

---

## Project structure
```bash
lstm-financial-prediction/
│
├── src/
│ ├── data.py # Data download and preprocessing (yfinance)
│ ├── model.py # Model builder and custom loss
│ ├── train.py # Training script (per ticker)
│ ├── evaluate.py # Evaluation + visualization
│ └── utils.py # small helpers (save/load)
│
├── notebooks/ # Original exploratory notebook (optional)
├── models/ # Saved Keras model files (.h5) - ignored by git
├── results/ # Saved plots and metrics - ignored by git
├── requirements.txt
├── .gitignore
└── LICENSE
```
---
