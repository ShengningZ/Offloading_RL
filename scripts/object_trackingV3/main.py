# main.py
import cv2
import grpc
import numpy as np
import project_data_pb2
import project_data_pb2_grpc
from model_loader import load_yolo_model
from object_detection import detect_objects
from kalman_filter_module import KalmanFilter, apply_kalman_filter
from background_subtraction import apply_background_subtraction
from visualization import visualize_mask, visualize_detections, visualize_kalman_prediction
from utilities import filter_detections_by_mask
from kalman_utils import kalman_predict_and_update, update_kalman_with_contours, adjust_kalman_parameters

def should_offload_operation(operation):
    # Placeholder for logic to decide whether to offload processing
    # Could be based on network conditions, device load, etc.
    return False

def main():
    # Initialize components
    yolo_model = load_yolo_model()
    kf = KalmanFilter()
    
    cap = cv2.VideoCapture(0)  # Assume using the webcam
    
    # Create gRPC channels to edge devices
    background_subtraction_channel = grpc.insecure_channel('localhost:50051')
    object_detection_channel = grpc.insecure_channel('localhost:50052')
    kalman_filter_channel = grpc.insecure_channel('localhost:50053')

    # Create gRPC stubs
    background_subtraction_stub = project_data_pb2_grpc.BackgroundSubtractionServiceStub(background_subtraction_channel)
    object_detection_stub = project_data_pb2_grpc.ObjectDetectionServiceStub(object_detection_channel)
    kalman_filter_stub = project_data_pb2_grpc.KalmanFilterServiceStub(kalman_filter_channel)
    
    last_measurement = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        offload_background_subtraction = should_offload_operation("background_subtraction")
        fg_mask_thresh = apply_background_subtraction(background_subtraction_stub, frame, offload=offload_background_subtraction)
        visualize_mask(fg_mask_thresh)
        
        offload_object_detection = should_offload_operation("object_detection")
        detections = detect_objects(object_detection_stub, frame, yolo_model, offload=offload_object_detection)
        
        filtered_detections = filter_detections_by_mask(detections, fg_mask_thresh)
        visualize_detections(frame, filtered_detections)
        
        offload_kalman_filter = should_offload_operation("kalman_filter")
        predicted_state, velocity = apply_kalman_filter(kalman_filter_stub, filtered_detections, kf, offload=offload_kalman_filter)
        
        # Continuously improve Kalman filter predictions
        if last_measurement is not None:
            adjust_kalman_parameters(kf, last_measurement, np.array([predicted_state[0][0], predicted_state[1][0]]))
        last_measurement = np.array([predicted_state[0][0], predicted_state[1][0]])
        
        update_kalman_with_contours(kf, frame, background_subtraction_stub)
        predicted_state = kalman_predict_and_update(kf, filtered_detections)
        
        visualize_kalman_prediction(frame, predicted_state, velocity)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()