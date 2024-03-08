import cv2

# Function to capture and save an image from the webcam
def capture_image_from_webcam(image_path='webcam_image.jpg'):
    # Initialize the webcam (0 by default)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Save the captured image to a file
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")
    else:
        print("Failed to capture image")
    
    # Release the webcam
    cap.release()

# Capture and save an image
capture_image_from_webcam()