import numpy as np

class DriftDetector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.drift_points = []

    def detect(self, reconstruction_errors):
        for i, err in enumerate(reconstruction_errors):
            if err > self.threshold:
                self.drift_points.append(i)
        return self.drift_points
