import numpy as np

def create_sliding_windows(data, window_size, step=1):
    """
    Create sliding windows from input data.
    
    Args:
        data: numpy array of shape (n_samples, n_features) or (n_samples,)
        window_size: int, size of the sliding window
        step: int, step size for sliding window (default=1)
    
    Returns:
        numpy array of shape (n_windows, window_size, n_features) or (n_windows, window_size)
    """
    # Convert input to numpy array if it isn't already
    data = np.asarray(data)
    
    # Validate inputs
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if len(data) < window_size:
        raise ValueError("window_size is larger than the number of samples")
    
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]
        windows.append(window)
    
    return np.array(windows)