import sys
import os

print("Initial sys.path:", sys.path)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
print("Updated sys.path:", sys.path)

try:
    from preprocessing.windowing import create_sliding_windows
    print("Imported create_sliding_windows:", create_sliding_windows)
except ImportError as e:
    print("ImportError:", e)
    import preprocessing.windowing
    print("Available attributes in windowing:", dir(preprocessing.windowing))
    raise

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from model.lstm_autoencoder import LSTMAutoencoder
from drift.drift_detector import DriftDetector
from evaluation.metrics import calculate_reconstruction_error
from app.config.settings import settings

df = pd.read_csv("data/raw/mixed_1010_abrupto.csv")
X = df.iloc[:, :-1].values  # Features: X1, X2, X3, X4
y = df['class'].values      # Labels for validation

# Simulate preprocessing
X_windows = create_sliding_windows(X, window_size=10)
X_tensor = torch.tensor(X_windows).float()

# Load model
model = LSTMAutoencoder(input_dim=X.shape[1], hidden_dim=64)
model.load_state_dict(torch.load(settings.MODEL_PATH))
model.eval()

# Run inference
with torch.no_grad():
    recon = model(X_tensor)
    errors = calculate_reconstruction_error(X_tensor.numpy(), recon.numpy())

# Print error statistics
print("Reconstruction errors:", errors)
print("Mean error:", errors.mean(), "Max error:", errors.max())
print("Threshold:", settings.THRESHOLD)

# Detect drift
detector = DriftDetector(threshold=settings.THRESHOLD)
drift_points = detector.detect(errors)

#Convert drift points to binary predictions

y_pred = np.zeros(len(y_true, dtype))
y_true_adj = y_true[:10] if len(y_true) >10 else y_true

print("Drift detected at indices:", drift_points)
print("True drift points (class=1) indices:", np.where(y[10:] == 1)[0])  # Adjust for windowing

# Plot errors and true drift
plt.figure(figsize=(12, 6))
plt.plot(errors, label='Reconstruction Errors')
plt.plot(np.where(y[10:] == 1)[0], [errors[i] for i in np.where(y[10:] == 1)[0]], 'ro', label='True Drift (class=1)')
plt.axhline(y=settings.THRESHOLD, color='r', linestyle='--', label=f'Threshold={settings.THRESHOLD}')
plt.title("Reconstruction Errors with True Drift Points")
plt.legend()
plt.show()