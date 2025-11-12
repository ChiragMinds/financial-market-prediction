# src/utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

def save_model(model, path):
    """
    Save Keras model to disk (HDF5) and print confirmation.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f" Model saved to {path}")

def save_history(history, path):
    """
    Save training history dictionary (Keras History.history) to numpy file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array(history, dtype=object))
    print(f" History saved to {path}")

def plot_loss(history, save_path):
    """Plot training and validation loss from a Keras History object."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history.get('loss', []), label='train')
    plt.plot(history.history.get('val_loss', []), label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" Loss plot saved to {save_path}")

def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f" Scaler saved to {path}")

def load_scaler(path):
    return joblib.load(path)
