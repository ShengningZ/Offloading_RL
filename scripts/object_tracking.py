import cv2
import numpy as np

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

# 封装物体检测步骤
def step_detect_objects(frame, object_detector):
    detections = detect_objects(frame, object_detector)
    return detections

def detect_objects(frame, object_detector):
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    return valid_contours

# 封装预测步骤
def step_predict_state(kf):
    predicted_state = kf.predict()
    print("Step 2: Predicting the next state")
    return predicted_state

# 封装更新步骤
def step_update_state(kf, measurement):
    updated_state = kf.correct(measurement)
    print("Step 3: Updating the state with the new measurement")
    return updated_state

# 封装可视化步骤
def step_visualize_results(frame, detections, contours, predicted_state, kf, trail_length=10):
    frame_with_detections = np.zeros_like(frame)
    cv2.drawContours(frame_with_detections, contours, -1, (255, 255, 255), 1)
    print("Step 4: Visualizing the results")
    frame_with_prediction = frame.copy()
    
    # Ensure predicted_x and predicted_y are defined
    predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
    
    # Draw predicted positions based on current detections and velocity
    if kf:
        velocity_x, velocity_y = int(kf.kf.statePost[2]), int(kf.kf.statePost[3])
        # Draw future positions based on velocity
        for i in range(1, trail_length + 1):
            future_pos_x = predicted_x + (velocity_x * i)
            future_pos_y = predicted_y + (velocity_y * i)
            cv2.circle(frame_with_prediction, (future_pos_x, future_pos_y), 1, (0, 255, 255), -1)
        
        # Draw direction arrow using the velocity
        cv2.arrowedLine(frame_with_prediction, (predicted_x, predicted_y), 
                        (predicted_x + velocity_x * 5, predicted_y + velocity_y * 5), 
                        (0, 0, 255), 2)
    
    cv2.imshow("Detections Mask", frame_with_detections)
    cv2.imshow("Kalman Prediction", frame_with_prediction)

def main():
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    kf = KalmanFilter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        contours = detect_objects(frame, object_detector)
        predicted_state = step_predict_state(kf)

        detections = []  # List to hold the centers of detected objects
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detections.append((cx, cy))
        
        if detections:
            # Assuming the first detection for simplification
            measurement = np.array([[np.float32(detections[0][0])], [np.float32(detections[0][1])]])
            step_update_state(kf, measurement)

        step_visualize_results(frame, detections, contours, predicted_state, kf)

        if cv2.waitKey(30) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()