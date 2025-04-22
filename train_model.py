# c:\ai_drift_detection\train_model.py
import torch
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model.lstm_autoencoder import LSTMAutoencoder
from preprocessing.windowing import create_sliding_windows
from model.trainer import train_model

# Load data
df = pd.read_csv("data/raw/mixed_1010_abrupto.csv")
X = df.iloc[:, :-1].values
X_windows = create_sliding_windows(X, window_size=10)
X_tensor = torch.tensor(X_windows).float()

# Create DataLoader
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = LSTMAutoencoder(input_dim=X.shape[1], hidden_dim=64)

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, dataloader, n_epochs=50, lr=0.001, device=device, save_path="models/lstm_autoencoder.pth")