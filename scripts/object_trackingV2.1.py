import cv2
import numpy as np
import torch
import time

def load_yolo_model():
    # Load the YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.random.randn(4, 1).astype(np.float32)

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        return self.kf.correct(measurement)
    

# Detect Objects with YOLO
def detect_objects_yolo(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    detections = results.xyxy[0]
    return detections

# Update and Predict with Kalman Filter
def kalman_predict_and_update(kf, detections):
    if len(detections) > 0:
        for detection in detections:
            x1, y1, x2, y2, conf, _ = detection.cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            kf.correct(np.array([[cx], [cy]], dtype=np.float32))
    predicted_state = kf.predict()
    return predicted_state


def update_kalman_with_contours(kf, frame, object_detector):
    # Apply background subtraction
    fg_mask = object_detector.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is our object of interest
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Update Kalman filter with the centroid of the largest contour
            kf.correct(np.array([[cx], [cy]], dtype=np.float32))

def adjust_kalman_parameters(kf, last_measurement, current_measurement):
    # Calculate velocity or the difference between predictions and measurements
    velocity = np.linalg.norm(current_measurement - last_measurement)

    # Adjust process noise based on velocity or accuracy
    if velocity > 10:
        kf.processNoiseCov = np.eye(4) * 0.5
    else:
        kf.processNoiseCov = np.eye(4) * 1e-3

# Function to visualize the foreground mask
def visualize_mask(fg_mask):
    if fg_mask is not None:
        cv2.imshow("Foreground Mask", fg_mask)

# Function to visualize YOLO detections
def visualize_yolo_detections(frame, detections):
    frame_detections = frame.copy()
    for detection in detections:
        x1, y1, x2, y2, conf, _ = detection.cpu().numpy()
        cv2.rectangle(frame_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imshow("YOLO Detections", frame_detections)

# Function to visualize Kalman prediction with an arrow
def visualize_kalman_prediction(frame, predicted_state, kf):
    frame_prediction = frame.copy()
    predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
    velocity_x, velocity_y = int(kf.kf.statePost[2][0]), int(kf.kf.statePost[3][0])

    cv2.circle(frame_prediction, (predicted_x, predicted_y), 10, (0, 0, 255), -1)
    cv2.arrowedLine(frame_prediction, (predicted_x, predicted_y),
                    (predicted_x + velocity_x * 5, predicted_y + velocity_y * 5),
                    (255, 0, 0), 2)
    cv2.imshow("Kalman Prediction", frame_prediction)

def find_object_within_movement(detections, contours):
    for detection in detections:
        x1, y1, x2, y2, conf, _ = detection.cpu().numpy()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        for contour in contours:
            if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                return detection  # Returns the first detection within a movement area
    return None

def visualize_results(frame, detections, predicted_state, kf, fg_mask):
    """
    Visualizes the results including the mask, YOLO detections, and Kalman prediction.
    """
    # Visualize the mask
    visualize_mask(fg_mask)

    # Visualize YOLO detections
    visualize_yolo_detections(frame, detections)

    # Visualize Kalman prediction with an arrow if a predicted state is available
    if predicted_state is not None:
        visualize_kalman_prediction(frame, predicted_state, kf)
        
def process_prediction_and_visualization(frame, object_detector, yolo_model, kf):
    """
    Processes the frame to identify the area of movement, match YOLO detections to it,
    and then update and predict with the Kalman Filter.
    """
    fg_mask = object_detector.apply(frame)
    _, fg_mask_thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualize the mask
    visualize_mask(fg_mask_thresh)

    yolo_detections = detect_objects_yolo(frame, yolo_model)
    filtered_detections = filter_detections_by_mask(yolo_detections, fg_mask_thresh)

    # Visualize YOLO detections after filtering
    visualize_yolo_detections(frame, filtered_detections)

    if filtered_detections:
        selected_detection = filtered_detections[0]
        x1, y1, x2, y2, conf, _ = selected_detection.cpu().numpy()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        current_position = [cx, cy]
        kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        predicted_state = kf.predict()
        # Visualize Kalman prediction with an arrow
        visualize_kalman_prediction(frame, predicted_state, kf)
        return predicted_state, current_position
    else:
        predicted_state = kf.predict()
        visualize_kalman_prediction(frame, predicted_state, kf)
        return predicted_state, None


def filter_detections_by_mask(detections, fg_mask):
    """
    Filters detections based on the foreground mask.

    Parameters:
    - detections: List of detections from YOLO.
    - fg_mask: Foreground mask obtained from background subtraction.

    Returns:
    - filtered_detections: Detections that are within the foreground areas.
    """
    filtered_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # If the center of the bounding box is within a foreground area, add it to the filtered detections
        if fg_mask[center_y, center_x] > 0:
            filtered_detections.append(detection)

    return filtered_detections

def main():
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    yolo_model = load_yolo_model()
    kf = KalmanFilter()
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    
    last_position = None
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        time_elapsed = current_time - last_time

        # Process each frame for prediction and visualization
        predicted_state, current_position = process_prediction_and_visualization(frame, object_detector, yolo_model, kf)

        if current_position is not None and last_position is not None:
            # Calculate speed in pixels per second
            distance = np.linalg.norm(np.array(current_position) - np.array(last_position))
            speed = distance / time_elapsed
            print(f"Speed: {speed:.2f} pixels/second")

        last_position = current_position if current_position is not None else last_position
        last_time = current_time

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()