import cv2
import numpy as np
# Import the generated protobuf classes
from kalman_filter_pb2 import State, StateUpdate

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

    def export_state_to_protobuf(self):
        current_state = self.kf.statePost.ravel()  # Flatten the state to a 1D array
        state_message = State(x=current_state[0], y=current_state[1], vx=current_state[2], vy=current_state[3])
        return state_message

def main():
    # Initialize the Kalman Filter with an example initial state
    kf = KalmanFilter(initial_state=[1, 2, 0.5, 1])

    # Simulate some measurements for correcting the Kalman Filter
    measurements = [
        [2, 4],
        [3, 6],
        [4, 8],
        [5, 10]
    ]

    # Iterate over the measurements, correct the Kalman Filter, and export the state
    for measurement in measurements:
        kf.predict()
        kf.correct(measurement)
        exported_state = kf.export_state_to_protobuf()
        print(f"Exported State: x={exported_state.x}, y={exported_state.y}, vx={exported_state.vx}, vy={exported_state.vy}")

if __name__ == "__main__":
    main()