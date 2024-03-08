# object_detection.py
import cv2
import torch
import numpy as np
import project_data_pb2
import project_data_pb2_grpc

def detect_objects_yolo(frame, model, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 640))  # Resize the frame to match the model's input size
    frame_transposed = frame_resized.transpose((2, 0, 1))  # Change the shape from (H, W, C) to (C, H, W)
    frame_tensor = torch.from_numpy(frame_transposed).to(device).float() / 255.0  # Normalize pixel values
    if len(frame_tensor.shape) == 3:
        frame_tensor = frame_tensor.unsqueeze(0)
    
    model.eval()
    
    with torch.no_grad():
        results = model(frame_tensor)
    
    detections = []
    if isinstance(results, torch.Tensor):
        detections = results.cpu().numpy().tolist()
    elif isinstance(results, list):
        detections = [det.cpu().numpy().tolist() for det in results]
    elif hasattr(results, 'xyxy'):
        detections = results.xyxy[0].cpu().numpy().tolist()
    elif hasattr(results, 'xywh'):
        detections = results.xywh[0].cpu().numpy().tolist()
    else:
        raise ValueError("Unsupported model output format. Please check your model and version.")

    return detections

def detect_objects(stub, frame, model=None, offload=False):
    if offload:
        image_proto = project_data_pb2.Image(
            width=frame.shape[1], height=frame.shape[0],
            format='BGR', data=cv2.imencode('.jpg', frame)[1].tobytes()
        )
        detection_results = stub.DetectObject(image_proto)
        detections = [
            [det.x1, det.y1, det.x2, det.y2, det.confidence, det.label]
            for det in detection_results.detections
        ]
    else:
        detections = detect_objects_yolo(frame, model)
    
    return detections