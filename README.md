<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange.svg" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/License-All--Rights--Reserved-red.svg" alt="License" />
  <img src="https://img.shields.io/github/stars/<your-username>/lstm-financial-prediction?style=social" alt="GitHub stars" />
</p>

<h1 align="center">📈 LSTM Networks for Financial Market Prediction</h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/<your-username>/lstm-financial-prediction/main/docs/hero.png" alt="Project hero image" width="800" />
</p>

---

## Project summary
This repository contains a modular conversion of an experimental notebook into a reproducible project that trains sequence models for short-term financial market prediction. The code downloads historical price data for four tickers (`AAPL`, `GOOG`, `MSFT`, `AMZN`) using `yfinance`, prepares sliding-window sequences, trains a deep neural network (Conv1D + BiLSTM + attention) per ticker, and evaluates performance using multiple metrics (MAE, RMSE, MAPE, directional accuracy).

This repository is intended for **demonstration and academic use**. The code, models, and outputs are organized for clarity and reproducibility.

---

## 🔧 Features and components
- Internal data fetching with `yfinance` (no local dataset required).  
- Data preprocessing and sliding-window sequence creation per ticker.  
- Modular model builder (Conv1D + Bidirectional LSTM + Attention).  
- Training script with callbacks, per-ticker model saving, and loss plots.  
- Evaluation script that produces numeric metrics and forecast visualizations.  
- Structured outputs: trained models saved in `models/`, plots in `results/`.

---

## 🧰 Project structure
```bash
lstm-financial-prediction/
│
├── src/
│   ├── data.py           # Data download and preprocessing (yfinance)
│   ├── model.py          # Model builder and custom loss
│   ├── train.py          # Training script (per ticker)
│   ├── evaluate.py       # Evaluation + visualization
│   └── utils.py          # small helpers (save/load)
│
├── notebooks/            # Original exploratory notebook (optional)
├── models/               # Saved Keras model files (.h5) - ignored by git
├── results/              # Saved plots and metrics - ignored by git
├── requirements.txt
├── .gitignore
└── LICENSE

```
---
## Notes on reproducibility and usage

- **Data fetching**: This repository downloads data at runtime using `yfinance`. Internet access is required for data fetching. If `yfinance` fails (rate limits or connection errors), retry or use a local CSV export and point the scripts to local files.

- **Saved models**: Trained models are saved as **Keras `.h5`** files under the `models/` folder. These contain model weights and architecture.

- **Scalers**: Scalers are fit **per ticker** during preprocessing (MinMax scaling). By default scalers are created in-memory and used during training and inference. If you want scalers persisted so predictions can be inverse-transformed outside the training session, enable the scaler saving code (example provided in `src/utils.py`) — saved scalers are portable via `joblib`.

- **Loss function**: The training loss is a custom loss that combines **MSE** with a directional penalty (to encourage correct direction of multi-step forecasts). The model is compiled with that custom loss during training.

- **Reproducibility**: To reproduce results, ensure:
  - same `--start` / `--end` date ranges when fetching data with `yfinance`,
  - identical model and training hyperparameters,
  - same TensorFlow / Python versions (see `requirements.txt` or the environment used in the notebook),
  - fix random seeds for NumPy / TensorFlow if deterministic runs are required.

- **Hardware**: The model uses stacked BiLSTM layers; CPU training can be slow. For practical experiments use a GPU-enabled runtime (Colab + GPU or local GPU with the correct TensorFlow wheel).

---

## Authors and contacts

- **Chirag Chauhan** — primary contact: <chiragchauhan1401@gmail.com>  
- Collaborators: Himanshi Borad, Dhvani Maktuporia, Mayuri A. Mehta, Dheeraj Kumar Singh

If you’d like to reproduce experiments or request permission to reuse the code, please contact the primary author.

---

## License

All Rights Reserved © 2025 Chirag Chauhan.

This repository is provided for academic demonstration and evaluation only. No part of this code may be reused, redistributed, or modified without prior written permission from the author. To request licensing or reuse, contact Chirag Chauhan (chiragchauhan1401@gmail.com).
