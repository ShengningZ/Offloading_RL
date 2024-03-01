import cv2

def initialize_object_detector():
    return cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

def apply_object_detector(detector, frame):
    fg_mask = detector.apply(frame)
    _, fg_mask_thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
    return fg_mask_thresh