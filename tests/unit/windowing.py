import numpy as np
from src.preprocessing.windowing import create_sliding_windows

def test_windowing():
    data = np.arange(10).reshape(10, 1)
    windows = create_sliding_windows(data, 3)
    assert windows.shape == (8, 3, 1)
def create_sliding_windows(data, window_size, step=1):
    """
    Splits data into overlapping sliding windows.
    Args:
        data: 2D or 3D numpy array
        window_size: number of time steps in each window
        step: step size for the sliding window
    Returns:
        windows: 3D numpy array of shape (num_windows, window_size, features)
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)