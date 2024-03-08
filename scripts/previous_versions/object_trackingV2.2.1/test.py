from model_loader import load_yolo_model
from object_detection import detect_objects_yolo
from kalman_filter_module import KalmanFilter
from background_subtraction import initialize_object_detector, apply_object_detector
from visualization import visualize_mask, visualize_detections, visualize_kalman_prediction
from utilities import filter_detections_by_mask
import cv2
import numpy as np

def main():
    # Initialize the YOLO model
    yolo_model = load_yolo_model()
    
    # Initialize Kalman Filter
    kf = KalmanFilter()
    
    # Initialize Background Subtractor
    object_detector = initialize_object_detector()
    
    # Video capture
    video_source = 0  # Use 0 for webcam
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply Background Subtraction
        fg_mask_thresh = apply_object_detector(object_detector, frame)
        visualize_mask(fg_mask_thresh)

        # Detect Objects with YOLO
        detections = detect_objects_yolo(frame, yolo_model)

        # Filter detections based on the foreground mask
        filtered_detections = filter_detections_by_mask(detections, fg_mask_thresh)
        visualize_detections(frame, filtered_detections)

        # Update and predict with Kalman Filter
        if filtered_detections:
            for detection in filtered_detections:
                x1, y1, x2, y2, conf, _ = detection.cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                kf.correct(np.array([[cx], [cy]], dtype=np.float32))

        predicted_state = kf.predict()
        visualize_kalman_prediction(frame, predicted_state, kf.kf.statePost[2:4])

        # Handle frame rate and exit condition
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()