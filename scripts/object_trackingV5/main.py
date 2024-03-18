from modules.model_loader import load_yolo_model
from modules.object_detection import detect_objects_yolo
from modules.kalman_filter_module import KalmanFilter
from modules.background_subtraction import initialize_object_detector, apply_object_detector
from modules.visualization import visualize_mask, visualize_detections, visualize_kalman_prediction
from modules.utilities import filter_detections_by_mask
from modules.kalman_utils import kalman_predict_and_update, update_kalman_with_contours, adjust_kalman_parameters
import cv2
import numpy as np
import torch

def should_offload_operation():
    # Placeholder for logic to decide whether to offload processing
    # Could be based on network conditions, device load, etc.
    return False

def convert_detections_to_tensor(detections):
    return [torch.tensor([d[0], d[1], d[2], d[3], d[4], 0]) for d in detections]

def main():
    # Initialize components
    yolo_model = load_yolo_model()
    kf = KalmanFilter()
    object_detector = initialize_object_detector()
    
    cap = cv2.VideoCapture(0)  # Assume using the webcam
    last_measurement = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fg_mask_thresh = apply_object_detector(object_detector, frame)
        visualize_mask(fg_mask_thresh)
        
        if should_offload_operation():
            # Offload object detection (this is a placeholder, actual offloading implementation needed)
            detections = offload_object_detection(frame)
        else:
            # Local processing
            detections = detect_objects_yolo(frame, yolo_model)
        
        filtered_detections = filter_detections_by_mask(detections, fg_mask_thresh)
        filtered_detections = convert_detections_to_tensor(filtered_detections)
        visualize_detections(frame, filtered_detections)
        
        # Update Kalman filter based on detections
        predicted_state = kalman_predict_and_update(kf, filtered_detections)
        
        # Update Kalman filter based on contours
        update_kalman_with_contours(kf, frame, object_detector)
        
        # Adjust Kalman filter parameters based on object velocity
        current_measurement = kf.kf.statePost[:2]
        if last_measurement is not None:
            adjust_kalman_parameters(kf, last_measurement, current_measurement)
        last_measurement = current_measurement
        
        visualize_kalman_prediction(frame, predicted_state, kf.kf.statePost[2:4])

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()