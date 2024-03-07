#object_detection.py

import cv2
import torch

def detect_objects_yolo(frame, model, device='cuda'):
    """
    Detect objects in an image frame using a YOLO model.

    Parameters:
    - frame: The image frame in which to detect objects.
    - model: The YOLO model used for object detection.
    - device: The device to perform the detection on ('cpu' or 'cuda'). Default is 'cuda'.
    
    Returns:
    - Detections made by the model in the given frame.
    """
    # Check if CUDA is available, fallback to CPU if necessary or if explicitly requested
    if not torch.cuda.is_available() or device == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda'
    
    # Print the device being used for the user's information
    print(f"Performing object detection on: {device.upper()}")

    # Convert frame to the correct format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Process the frame as needed by the model (without unnecessary conversion to tensor if not needed)
    results = model(frame_rgb)

    # Adapt the handling of the model's output as per the expected structure
    if hasattr(results, 'xyxy'):
        # This means we have a Results object from Ultralytics which contains the xyxy attribute
        detections = results.xyxy[0]  # Assuming first batch item for simplicity
    else:
        # Handle other cases or raise an error/warning
        raise ValueError("Unexpected model output format, check the model and version being used.")

    # Assuming detections are already on CPU after accessing .xyxy[0]
    return detections