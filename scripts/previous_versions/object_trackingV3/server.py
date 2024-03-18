import sys
import os

# 获取当前脚本的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加 communication 目录到 sys.path
communication_dir = os.path.join(current_dir, 'communication')
sys.path.append(communication_dir)

import grpc
from concurrent import futures
import cv2
import numpy as np
import torch
from modules.background_subtraction import initialize_object_detector, apply_object_detector
from modules.object_detection import detect_objects_yolo
from modules.utilities import filter_detections_by_mask
from modules.kalman_filter_module import KalmanFilter
from modules.kalman_utils import kalman_predict_and_update, update_kalman_with_contours, adjust_kalman_parameters
import communication.project_data_pb2_grpc
import communication.project_data_pb2
import pdb
class BackgroundSubtractionServicer(communication.project_data_pb2_grpc.BackgroundSubtractionServiceServicer):
    def __init__(self):
        self.object_detector = initialize_object_detector()

    def ApplyBackgroundSubtraction(self, request, context):
        frame = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        fg_mask_thresh = apply_object_detector(self.object_detector, frame)
        fg_mask_data = cv2.imencode('.png', fg_mask_thresh)[1].tobytes()
        return communication.project_data_pb2.ForegroundMask(mask_data=fg_mask_data, width=fg_mask_thresh.shape[1], height=fg_mask_thresh.shape[0])

class ObjectDetectionServicer(communication.project_data_pb2_grpc.ObjectDetectionServiceServicer):
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def DetectObjects(self, request, context):
        frame = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        detections = detect_objects_yolo(frame, self.model)
        detection_results = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            detection_results.append(communication.project_data_pb2.Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label=str(cls)))
        return communication.project_data_pb2.DetectionResult(detections=detection_results)

class FilteringServicer(communication.project_data_pb2_grpc.FilteringServiceServicer):
    def FilterDetectionsByMask(self, request, context):
        detections = [(d.x1, d.y1, d.x2, d.y2, d.confidence, d.label) for d in request.detection_result.detections]
        fg_mask = cv2.imdecode(np.frombuffer(request.foreground_mask.mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        filtered_detections = filter_detections_by_mask(detections, fg_mask)
        filtered_detection_results = []
        for detection in filtered_detections:
            x1, y1, x2, y2, conf, cls = detection
            filtered_detection_results.append(communication.project_data_pb2.Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label=cls))
        return communication.project_data_pb2.DetectionResult(detections=filtered_detection_results)

class KalmanFilterServicer(communication.project_data_pb2_grpc.KalmanFilterServiceServicer):
    def __init__(self):
        self.kf = KalmanFilter()

    def UpdateState(self, request, context):
        print("Received Kalman filter request:")
        print(request)

        detections = [(d.x1, d.y1, d.x2, d.y2, d.confidence, d.label) for d in request.detection_result.detections]
        print("Deserialized detections:")
        print(detections)

        predicted_state, state_post = self.local_kalman_filter(detections)

        return communication.project_data_pb2.StateUpdate(
            state=communication.project_data_pb2.State(
                x=float(predicted_state[0][0]),
                y=float(predicted_state[1][0]),
                vx=float(predicted_state[2][0]),
                vy=float(predicted_state[3][0])
            ),
            state_post=communication.project_data_pb2.StatePost(
                x=float(state_post[0][0]),
                y=float(state_post[1][0]),
                vx=float(state_post[2][0]),
                vy=float(state_post[3][0])
            )
        )

    def local_kalman_filter(self, detections):
        for detection in detections:
            x1, y1, x2, y2, conf, _ = detection
            self.kf.correct(np.array([[(x1+x2)/2], [(y1+y2)/2]], dtype=np.float32))
        predicted_state = self.kf.predict()
        state_post = self.kf.kf.statePost
        return predicted_state, state_post

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    communication.project_data_pb2_grpc.add_BackgroundSubtractionServiceServicer_to_server(BackgroundSubtractionServicer(), server)
    communication.project_data_pb2_grpc.add_ObjectDetectionServiceServicer_to_server(ObjectDetectionServicer(), server)
    communication.project_data_pb2_grpc.add_FilteringServiceServicer_to_server(FilteringServicer(), server)
    communication.project_data_pb2_grpc.add_KalmanFilterServiceServicer_to_server(KalmanFilterServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()