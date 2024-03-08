# kalman_filter_module.py
import cv2
import numpy as np
import project_data_pb2
import project_data_pb2_grpc

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

def apply_kalman_filter(stub, detections, kf=None, offload=False):
    if offload:
        detection_results = project_data_pb2.DetectionResult(detections=[
            project_data_pb2.Detection(
                x1=float(det[0]), y1=float(det[1]), x2=float(det[2]), y2=float(det[3]),
                confidence=float(det[4]), label=str(det[5])
            ) for det in detections
        ])
        state_update = stub.UpdateState(detection_results)
        predicted_state = np.array([[state_update.state.x], [state_update.state.y]])
        velocity = np.array([[state_update.state.vx], [state_update.state.vy]])
    else:
        if kf is None:
            kf = KalmanFilter()
        for detection in detections:
            kf.correct(np.array([[(detection[0] + detection[2]) / 2], [(detection[1] + detection[3]) / 2]], dtype=np.float32))
        predicted_state = kf.predict()
        velocity = kf.kf.statePost[2:4]
    
    return predicted_state, velocity