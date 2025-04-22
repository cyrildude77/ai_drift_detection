import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing.windowing import create_sliding_windows
import numpy as np

print("Function imported successfully!")
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
windows = create_sliding_windows(data, window_size=3)
print("Windows shape:", windows.shape)
print(windows)