from model_loader import load_yolo_model
from object_detection import detect_objects_yolo
from background_subtraction import initialize_object_detector
import cv2
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    yolo_model = load_yolo_model().to(device)
    
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)
    object_detector = initialize_object_detector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect Objects with YOLO (make sure to adapt detect_objects_yolo to handle device)
        detections = detect_objects_yolo(frame, yolo_model, device)
        
        # Skipping visualization for brevity, add if needed
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()