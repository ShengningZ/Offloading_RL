# background_subtraction.py
import cv2
import numpy as np
import project_data_pb2
import project_data_pb2_grpc

def initialize_object_detector(history=100, varThreshold=40, detectShadows=True):
    return cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)

def apply_object_detector(detector, frame, threshold=254):
    fg_mask = detector.apply(frame)
    _, fg_mask_thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
    return fg_mask_thresh

def apply_background_subtraction(stub, frame, threshold=254, offload=False):
    if offload:
        image_proto = project_data_pb2.Image(
            width=frame.shape[1], height=frame.shape[0],
            format='BGR', data=cv2.imencode('.jpg', frame)[1].tobytes()
        )
        fg_mask_proto = stub.ApplyBackgroundSubtraction(image_proto)
        fg_mask_thresh = cv2.imdecode(np.frombuffer(fg_mask_proto.mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        object_detector = initialize_object_detector()
        fg_mask_thresh = apply_object_detector(object_detector, frame, threshold)
    
    return fg_mask_thresh