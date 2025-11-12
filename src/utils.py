# src/utils.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def plot_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")
    else:
        plt.show()

def inverse_scale(scaler, arr):
    # arr shape: (n,1) or (1,)
    a = np.array(arr).reshape(-1,1)
    return scaler.inverse_transform(a).flatten()
