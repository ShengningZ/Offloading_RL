# kalman_utils.py

import cv2
import numpy as np

def kalman_predict_and_update(kf, detections):
    """
    Update the Kalman filter with object detections and predict the next state.

    Parameters:
    - kf: KalmanFilter instance.
    - detections: A list of detections, where each detection is a tensor of the form
                  [x1, y1, x2, y2, confidence, class_id].
    """
    if len(detections) > 0:
        for detection in detections:
            # Convert detection to CPU and extract coordinates
            x1, y1, x2, y2, conf, _ = detection.cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # Update the Kalman filter with the detection center
            kf.correct(np.array([[cx], [cy]], dtype=np.float32))
    
    # Predict the next state
    predicted_state = kf.predict()
    return predicted_state

def update_kalman_with_contours(kf, frame, object_detector):
    """
    Update the Kalman filter based on the largest contour found using background subtraction.

    Parameters:
    - kf: KalmanFilter instance.
    - frame: The current frame from the video feed.
    - object_detector: An initialized background subtractor object.
    """
    # Apply background subtraction
    fg_mask = object_detector.apply(frame)
    _, fg_mask_thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Update Kalman filter with the centroid of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            kf.correct(np.array([[cx], [cy]], dtype=np.float32))

def adjust_kalman_parameters(kf, last_measurement, current_measurement):
    """
    Adjust the Kalman filter's process noise covariance based on the velocity of the object.

    Parameters:
    - kf: KalmanFilter instance.
    - last_measurement: The last measurement point [x, y].
    - current_measurement: The current measurement point [x, y].
    """
    # Calculate velocity as the norm of the difference between measurements
    velocity = np.linalg.norm(current_measurement - last_measurement)

    # Adjust process noise covariance based on the object's velocity
    if velocity > 10:
        kf.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5
    else:
        kf.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
