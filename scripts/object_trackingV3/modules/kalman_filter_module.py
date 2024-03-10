#kalman_filter_module.py
import cv2
import numpy as np

class KalmanFilter:
    def __init__(self, initial_state=None, process_noise=0.03, measurement_noise=1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        if initial_state is not None:
            self.kf.statePost = np.array(initial_state, np.float32)
        else:
            self.kf.statePost = np.random.randn(4, 1).astype(np.float32)

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        return self.kf.correct(np.array(measurement, np.float32))

    def reset(self, initial_state=None, process_noise=None, measurement_noise=None):
        if initial_state is not None:
            self.kf.statePost = np.array(initial_state, np.float32)
        if process_noise is not None:
            self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        if measurement_noise is not None:
            self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise