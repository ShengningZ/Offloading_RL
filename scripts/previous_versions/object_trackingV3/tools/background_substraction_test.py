import cv2
from background_subtraction import initialize_object_detector, apply_object_detector

def test_background_subtraction(video_path):
    # Initialize the object detector with custom parameters
    detector = initialize_object_detector(history=150, varThreshold=50, detectShadows=False)

    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the object detector to get the foreground mask
        fg_mask_thresh = apply_object_detector(detector, frame, threshold=250)
        
        # Display the original frame and the foreground mask
        cv2.imshow('Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask_thresh)

        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_video_path = '/home/shengning/reinforcement rearning/scripts/object_trackingV2.2.1/output_video.avi'  # Update this path
    test_background_subtraction(test_video_path)