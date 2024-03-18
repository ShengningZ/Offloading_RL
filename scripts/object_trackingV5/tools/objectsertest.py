# main.py
from model_loader import load_yolo_model
from object_detection import detect_objects_yolo
import cv2
import torch
import numpy as np
import object_detection_pb2 # Generated from the Protobuf definition

def serialize_detections(detections):
    """Serialize detections to Protobuf format."""
    detection_result_proto = object_detection_pb2.DetectionResult()
    for det in detections:
        proto_det = detection_result_proto.detections.add()
        proto_det.x1, proto_det.y1, proto_det.x2, proto_det.y2, proto_det.confidence, proto_det.label = \
            det[0], det[1], det[2], det[3], det[4], str(det[5])
    return detection_result_proto.SerializeToString()

def deserialize_detections(serialized_data):
    """Deserialize Protobuf format detections back to list of detections."""
    detection_result_proto = object_detection_pb2.DetectionResult()
    detection_result_proto.ParseFromString(serialized_data)
    return [
        (det.x1, det.y1, det.x2, det.y2, det.confidence, det.label)
        for det in detection_result_proto.detections
    ]

def visualize_detections(frame, detections):
    """Visualize detection results on the frame."""
    for det in detections:
        x1, y1, x2, y2, confidence, label = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], det[5]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('Detections', frame)

def main():
    # Initialize the YOLO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = load_yolo_model(model_name='yolov5s', pretrained=True, device=device)
    
    cap = cv2.VideoCapture(0)  # Use the first webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        detections = detect_objects_yolo(frame, yolo_model, device)
        # Convert detections to list of tuples for serialization
        detections_list = [(det[0], det[1], det[2], det[3], det[4], det[5]) for det in detections]
        
        # Serialize and then deserialize detections
        serialized_data = serialize_detections(detections_list)
        deserialized_detections = deserialize_detections(serialized_data)
        
        # Visualize the results
        visualize_detections(frame, deserialized_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
