#mainwithremote.py
import sys
import os

# 获取当前脚本的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加 communication 目录到 sys.path
communication_dir = os.path.join(current_dir, 'communication')
sys.path.append(communication_dir)
import cv2
import numpy as np
import grpc
import torch
import communication.project_data_pb2 as project_data_pb2
import communication.project_data_pb2_grpc as project_data_pb2_grpc
from modules.background_subtraction import initialize_object_detector, apply_object_detector
from modules.object_detection import detect_objects_yolo
from modules.utilities import filter_detections_by_mask
from modules.kalman_filter_module import KalmanFilter
from modules.kalman_utils import kalman_predict_and_update
from modules.model_loader import load_yolo_model
from modules.visualization import visualize_mask, visualize_detections, visualize_kalman_prediction, visualize_filtered_detections

def local_background_subtraction(object_detector, frame):
    return apply_object_detector(object_detector, frame)

def remote_background_subtraction(stub, frame):
    frame_data = cv2.imencode('.jpg', frame)[1].tobytes()
    request = project_data_pb2.Image(data=frame_data, width=frame.shape[1], height=frame.shape[0], format='jpg')
    response = stub.ApplyBackgroundSubtraction(request)
    fg_mask_thresh = cv2.imdecode(np.frombuffer(response.mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return fg_mask_thresh

def local_object_detection(frame, model):
    return detect_objects_yolo(frame, model)

def remote_object_detection(stub, frame):
    frame_data = cv2.imencode('.jpg', frame)[1].tobytes()
    request = project_data_pb2.Image(data=frame_data, width=frame.shape[1], height=frame.shape[0], format='jpg')
    response = stub.DetectObjects(request)
    detections = [(d.x1, d.y1, d.x2, d.y2, d.confidence, d.label) for d in response.detections]
    return detections

def local_filtering(detections, fg_mask):
    return filter_detections_by_mask(detections, fg_mask)

def remote_filtering(stub, detections, fg_mask):
    fg_mask_data = cv2.imencode('.png', fg_mask)[1].tobytes()
    detections_proto = [project_data_pb2.Detection(x1=d[0], y1=d[1], x2=d[2], y2=d[3], confidence=d[4], label=str(int(float(d[5])))) for d in detections]
    request = project_data_pb2.FilteringRequest(
        detection_result=project_data_pb2.DetectionResult(detections=detections_proto),
        foreground_mask=project_data_pb2.ForegroundMask(mask_data=fg_mask_data, width=fg_mask.shape[1], height=fg_mask.shape[0])
    )
    response = stub.FilterDetectionsByMask(request)
    filtered_detections = [(d.x1, d.y1, d.x2, d.y2, d.confidence, d.label) for d in response.detections]
    return filtered_detections

def local_kalman_filter(kf, detections):
    for detection in detections:
        x1, y1, x2, y2, conf, _ = detection
        kf.correct(np.array([[(x1+x2)/2], [(y1+y2)/2]], dtype=np.float32))
    predicted_state = kf.predict()
    return predicted_state

def remote_kalman_filter(stub, filtered_detections):
    if len(filtered_detections) == 0:
        return None, None

    # 将过滤后的检测结果转换为 protobuf 消息格式
    detections_proto = [project_data_pb2.Detection(x1=d[0].item(), y1=d[1].item(), x2=d[2].item(), y2=d[3].item(), confidence=d[4].item(), label=str(int(d[5].item()))) for d in filtered_detections]

    # 创建 KalmanFilterRequest 消息
    kalman_request = project_data_pb2.KalmanFilterRequest(detection_result=project_data_pb2.DetectionResult(detections=detections_proto))

    # 调用远程卡尔曼滤波
    response = stub.UpdateState(kalman_request)

    predicted_state = np.array([response.state.x, response.state.y, response.state.vx, response.state.vy])
    state_post = np.array([response.state_post.x, response.state_post.y, response.state_post.vx, response.state_post.vy])

    return predicted_state, state_post

def convert_detections_to_tensor(detections):
    return [torch.tensor([d[0], d[1], d[2], d[3], d[4], 0]) for d in detections]

def main():
    # Initialize components
    yolo_model = load_yolo_model()
    kf = KalmanFilter()
    cap = cv2.VideoCapture(0)  # Assume using the webcam
    object_detector = initialize_object_detector()

    # Connect to gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    bg_subtraction_stub = project_data_pb2_grpc.BackgroundSubtractionServiceStub(channel)
    object_detection_stub = project_data_pb2_grpc.ObjectDetectionServiceStub(channel)
    filtering_stub = project_data_pb2_grpc.FilteringServiceStub(channel)
    kalman_filter_stub = project_data_pb2_grpc.KalmanFilterServiceStub(channel)

    # Offloading configuration
    offload_config = {
        'background_subtraction': True,
        'object_detection': False,
        'filtering': False,
        'kalman_filter': False
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if offload_config['background_subtraction']:
            fg_mask_thresh = remote_background_subtraction(bg_subtraction_stub, frame)
        else:
            fg_mask_thresh = local_background_subtraction(object_detector, frame)

        visualize_mask(fg_mask_thresh)

        if offload_config['object_detection']:
            detections = remote_object_detection(object_detection_stub, frame)
        else:
            detections = local_object_detection(frame, yolo_model).cpu().numpy().tolist()
            detections = [torch.tensor(d) for d in detections]  # Convert to list of PyTorch tensors

        if offload_config['filtering']:
            filtered_detections = remote_filtering(filtering_stub, detections, fg_mask_thresh)
        else:
            filtered_detections = local_filtering(detections, fg_mask_thresh)

        # Convert filtered_detections to a list of PyTorch tensors
        filtered_detections_tensors = convert_detections_to_tensor(filtered_detections)

        # Visualize detections
        visualize_detections(frame, filtered_detections_tensors)

        print(f"Type of filtered_detections: {type(filtered_detections)}")
        print(f"Filtered detections: {filtered_detections}")
        
        if offload_config['kalman_filter']:
            # Convert filtered detections to protobuf message format
            detections_proto = [project_data_pb2.Detection(x1=d[0], y1=d[1], x2=d[2], y2=d[3], confidence=d[4], label=str(int(d[5].split('.')[0]))) for d in filtered_detections]

            # Create KalmanFilterRequest message
            kalman_request = project_data_pb2.KalmanFilterRequest(detection_result=project_data_pb2.DetectionResult(detections=detections_proto))

            # Call remote Kalman filter
            predicted_state, state_post = remote_kalman_filter(kalman_filter_stub, kalman_request)
        else:
            predicted_state = local_kalman_filter(kf, filtered_detections)
            state_post = kf.kf.statePost

        print("Predicted state:", predicted_state)
        velocity = state_post[2:4]
        print(f"Velocity: vx={velocity[0]}, vy={velocity[1]}")

        if predicted_state is not None:
            visualize_kalman_prediction(frame, predicted_state, velocity)


        # visualize_detections(frame, convert_detections_to_tensor(filtered_detections))


        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()