import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2, axis=(1,2))
if __name__ == "__main__":
    print("metrics.py loaded correctly")
