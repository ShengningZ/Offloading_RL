#visualization.py

import cv2

def visualize_mask(fg_mask):
    cv2.imshow("Foreground Mask", fg_mask)

def visualize_detections(frame, detections, show_confidence=True):
    for detection in detections:
        x1, y1, x2, y2, conf, _ = detection.cpu().numpy()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        if show_confidence:
            text = f"{conf:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Detections", frame)
    
def visualize_kalman_prediction(frame, predicted_state, velocity):
    predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
    velocity_x, velocity_y = int(velocity[0]), int(velocity[1])
    cv2.circle(frame, (predicted_x, predicted_y), 10, (0, 0, 255), -1)
    cv2.arrowedLine(frame, (predicted_x, predicted_y), (predicted_x + velocity_x * 5, predicted_y + velocity_y * 5), (255, 0, 0), 2)
    cv2.imshow("Kalman Prediction", frame)