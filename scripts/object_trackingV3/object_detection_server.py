# object_detection_server.py
import grpc
from concurrent import futures
import project_data_pb2
import project_data_pb2_grpc
from model_loader import load_yolo_model
from object_detection import detect_objects_yolo
import cv2
import numpy as np

class ObjectDetectionServicer(project_data_pb2_grpc.ObjectDetectionServiceServicer):
    def __init__(self):
        self.model = load_yolo_model()

    def DetectObject(self, request, context):
        frame = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        detections = detect_objects_yolo(frame, self.model)
        detection_results = project_data_pb2.DetectionResult(detections=[
            project_data_pb2.Detection(
                x1=float(det[0]), y1=float(det[1]), x2=float(det[2]), y2=float(det[3]),
                confidence=float(det[4]), label=str(det[5])
            ) for det in detections
        ])
        return detection_results

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    project_data_pb2_grpc.add_ObjectDetectionServiceServicer_to_server(ObjectDetectionServicer(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()