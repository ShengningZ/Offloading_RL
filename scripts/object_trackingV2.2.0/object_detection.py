import cv2

def detect_objects_yolo(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    return results.xyxy[0]  # Returns detections
