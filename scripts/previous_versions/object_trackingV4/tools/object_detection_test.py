import cv2
from model_loader import load_yolo_model
from object_detection import detect_objects_yolo
import torch

def test_object_detection(device):
    print(f"Testing object detection on {device.upper()}...")

    # Load the YOLO model onto the specified device
    model = load_yolo_model(device=device)

    # Load a test image
    frame = cv2.imread('/home/shengning/reinforcement rearning/scripts/object_trackingV2.2.1/webcam_image.jpg')  # Update this path to your test image

    # Detect objects in the image
    detections = detect_objects_yolo(frame, model, device=device)

    # Check if detections were made
    if len(detections) > 0:
        print(f"Objects detected: {len(detections)}")
    else:
        print("No objects detected.")

if __name__ == "__main__":
    # Test on CPU
    test_object_detection('cpu')
    
    # Test on GPU, if available
    if torch.cuda.is_available():
        test_object_detection('cuda')