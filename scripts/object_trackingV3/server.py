import grpc
from concurrent import futures
import communication.project_data_pb2 as project_data_pb2
import communication.project_data_pb2_grpc as project_data_pb2_grpc
import cv2
import numpy as np
import torch
from modules.background_subtraction import initialize_object_detector, apply_object_detector
from modules.object_detection import detect_objects_yolo
from modules.utilities import filter_detections_by_mask
from modules.kalman_filter_module import KalmanFilter
from modules.kalman_utils import kalman_predict_and_update, update_kalman_with_contours, adjust_kalman_parameters

class BackgroundSubtractionServicer(project_data_pb2_grpc.BackgroundSubtractionServiceServicer):
    def __init__(self):
        self.object_detector = initialize_object_detector()

    def ApplyBackgroundSubtraction(self, request, context):
        frame = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        fg_mask_thresh = apply_object_detector(self.object_detector, frame)
        fg_mask_data = cv2.imencode('.png', fg_mask_thresh)[1].tobytes()
        return project_data_pb2.ForegroundMask(mask_data=fg_mask_data, width=fg_mask_thresh.shape[1], height=fg_mask_thresh.shape[0])

class ObjectDetectionServicer(project_data_pb2_grpc.ObjectDetectionServiceServicer):
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def DetectObjects(self, request, context):
        frame = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        detections = detect_objects_yolo(frame, self.model)
        detection_results = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            detection_results.append(project_data_pb2.Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label=str(cls)))
        return project_data_pb2.DetectionResult(detections=detection_results)

class FilteringServicer(project_data_pb2_grpc.FilteringServiceServicer):
    def FilterDetectionsByMask(self, request, context):
        detections = [project_data_pb2.Detection(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2, confidence=d.confidence, label=d.label) for d in request.detection_result.detections]
        fg_mask = cv2.imdecode(np.frombuffer(request.foreground_mask.mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        filtered_detections = filter_detections_by_mask(detections, fg_mask)
        filtered_detection_results = []
        for detection in filtered_detections:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            filtered_detection_results.append(project_data_pb2.Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label=str(cls)))
        return project_data_pb2.DetectionResult(detections=filtered_detection_results)

class KalmanFilterServicer(project_data_pb2_grpc.KalmanFilterServiceServicer):
    def __init__(self):
        self.kf = KalmanFilter()
        self.last_measurement = None

    def UpdateState(self, request, context):
        detections = [project_data_pb2.Detection(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2, confidence=d.confidence, label=d.label) for d in request.detections]
        
        # Update Kalman filter based on detections
        predicted_state = kalman_predict_and_update(self.kf, detections)

        # Update Kalman filter based on contours
        frame = cv2.imdecode(np.frombuffer(request.image.data, np.uint8), cv2.IMREAD_COLOR)
        update_kalman_with_contours(self.kf, frame, initialize_object_detector())

        # Adjust Kalman filter parameters based on object velocity
        current_measurement = self.kf.kf.statePost[:2]
        if self.last_measurement is not None:
            adjust_kalman_parameters(self.kf, self.last_measurement, current_measurement)
        self.last_measurement = current_measurement

        return project_data_pb2.StateUpdate(state=project_data_pb2.State(x=predicted_state[0], y=predicted_state[1], vx=predicted_state[2], vy=predicted_state[3]))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    project_data_pb2_grpc.add_BackgroundSubtractionServiceServicer_to_server(BackgroundSubtractionServicer(), server)
    project_data_pb2_grpc.add_ObjectDetectionServiceServicer_to_server(ObjectDetectionServicer(), server)
    project_data_pb2_grpc.add_FilteringServiceServicer_to_server(FilteringServicer(), server)
    project_data_pb2_grpc.add_KalmanFilterServiceServicer_to_server(KalmanFilterServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()