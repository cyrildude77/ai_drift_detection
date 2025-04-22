import pandas as pd
import numpy as np
from src.model.lstm_autoencoder import LSTMAutoencoder
from src.drift.drift_detector import DriftDetector

def detect_drift(file_bytes):
    df = pd.read_csv(file_bytes)
    data = df.iloc[:, :-1].values
    # TODO: scale + window
    # Load model
    model = LSTMAutoencoder(input_dim=data.shape[1], hidden_dim=64)
    model.load_state_dict(torch.load("models/lstm_model.pth"))
    model.eval()
    # TODO: run prediction + get reconstruction error
    # TODO: pass error to DriftDetector
    return "Sample Output"
