# kalman_filter_server.py
import grpc
from concurrent import futures
import project_data_pb2
import project_data_pb2_grpc
from kalman_filter_module import KalmanFilter
import numpy as np

class KalmanFilterServicer(project_data_pb2_grpc.KalmanFilterServiceServicer):
    def __init__(self):
        self.kf = KalmanFilter()

    def UpdateState(self, request, context):
        for detection in request.detections:
            self.kf.correct(np.array([[(detection.x1 + detection.x2) / 2], [(detection.y1 + detection.y2) / 2]], dtype=np.float32))
        
        predicted_state = self.kf.predict()
        state_update = project_data_pb2.StateUpdate(state=project_data_pb2.State(
            x=predicted_state[0], y=predicted_state[1],
            vx=self.kf.kf.statePost[2], vy=self.kf.kf.statePost[3]
        ))
        return state_update

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    project_data_pb2_grpc.add_KalmanFilterServiceServicer_to_server(KalmanFilterServicer(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()