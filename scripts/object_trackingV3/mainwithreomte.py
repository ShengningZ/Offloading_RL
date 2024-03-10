import cv2
import numpy as np
import grpc
import communication.project_data_pb2 as project_data_pb2
import communication.project_data_pb2_grpc as project_data_pb2_grpc
from modules.background_subtraction import initialize_object_detector, apply_object_detector
from modules.object_detection import detect_objects_yolo
from modules.utilities import filter_detections_by_mask
from modules.kalman_filter_module import KalmanFilter
from modules.kalman_utils import kalman_predict_and_update, update_kalman_with_contours, adjust_kalman_parameters
from modules.model_loader import load_yolo_model

def local_background_subtraction(frame):
    object_detector = initialize_object_detector()
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
    detections = [project_data_pb2.Detection(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2, confidence=d.confidence, label=d.label) for d in response.detections]
    return detections

def local_filtering(detections, fg_mask):
    return filter_detections_by_mask(detections, fg_mask)

def remote_filtering(stub, detections, fg_mask):
    fg_mask_data = cv2.imencode('.png', fg_mask)[1].tobytes()
    request = project_data_pb2.FilteringRequest(
        detection_result=project_data_pb2.DetectionResult(detections=detections),
        foreground_mask=project_data_pb2.ForegroundMask(mask_data=fg_mask_data, width=fg_mask.shape[1], height=fg_mask.shape[0])
    )
    response = stub.FilterDetectionsByMask(request)
    filtered_detections = [project_data_pb2.Detection(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2, confidence=d.confidence, label=d.label) for d in response.detections]
    return filtered_detections

def local_kalman_filter(kf, detections, frame):
    predicted_state = kalman_predict_and_update(kf, detections)
    update_kalman_with_contours(kf, frame, initialize_object_detector())
    current_measurement = kf.kf.statePost[:2]
    if kf.last_measurement is not None:
        adjust_kalman_parameters(kf, kf.last_measurement, current_measurement)
    kf.last_measurement = current_measurement
    return predicted_state

def remote_kalman_filter(stub, detections, frame):
    frame_data = cv2.imencode('.jpg', frame)[1].tobytes()
    request = project_data_pb2.DetectionResult(
        detections=detections,
        image=project_data_pb2.Image(data=frame_data, width=frame.shape[1], height=frame.shape[0], format='jpg')
    )
    response = stub.UpdateState(request)
    predicted_state = np.array([response.state.x, response.state.y, response.state.vx, response.state.vy])
    return predicted_state

def main():
    # Initialize components
    yolo_model = load_yolo_model()
    kf = KalmanFilter()
    cap = cv2.VideoCapture(0)  # Assume using the webcam

    # Connect to gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    bg_subtraction_stub = project_data_pb2_grpc.BackgroundSubtractionServiceStub(channel)
    object_detection_stub = project_data_pb2_grpc.ObjectDetectionServiceStub(channel)
    filtering_stub = project_data_pb2_grpc.FilteringServiceStub(channel)
    kalman_filter_stub = project_data_pb2_grpc.KalmanFilterServiceStub(channel)

    # Offloading configuration
    offload_config = {
        'background_subtraction': False,
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
            fg_mask_thresh = local_background_subtraction(frame)

        if offload_config['object_detection']:
            detections = remote_object_detection(object_detection_stub, frame)
        else:
            detections = local_object_detection(frame, yolo_model)

        if offload_config['filtering']:
            filtered_detections = remote_filtering(filtering_stub, detections, fg_mask_thresh)
        else:
            filtered_detections = local_filtering(detections, fg_mask_thresh)

        if offload_config['kalman_filter']:
            predicted_state = remote_kalman_filter(kalman_filter_stub, filtered_detections, frame)
        else:
            predicted_state = local_kalman_filter(kf, filtered_detections, frame)

        # Visualize the results
        # ...

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()