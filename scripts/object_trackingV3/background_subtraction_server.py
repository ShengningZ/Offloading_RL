# background_subtraction_server.py
import grpc
from concurrent import futures
import cv2
import numpy as np
import project_data_pb2
import project_data_pb2_grpc
from background_subtraction import initialize_object_detector, apply_object_detector

class BackgroundSubtractionServicer(project_data_pb2_grpc.BackgroundSubtractionServiceServicer):
    def __init__(self):
        self.object_detector = initialize_object_detector()

    def ApplyBackgroundSubtraction(self, request, context):
        frame = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        fg_mask = apply_object_detector(self.object_detector, frame)
        fg_mask_proto = project_data_pb2.ForegroundMask(
            mask_data=fg_mask.tobytes(),
            width=fg_mask.shape[1],
            height=fg_mask.shape[0]
        )
        return fg_mask_proto

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    project_data_pb2_grpc.add_BackgroundSubtractionServiceServicer_to_server(BackgroundSubtractionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()