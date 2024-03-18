import cv2

def record_video(video_path='output_video.avi', duration=10, frame_width=640, frame_height=480):
    # Define the codec and create VideoWriter object
    # The FourCC code is dependent on the format you want to record in
    # For an AVI file, using the MJPG codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))

    # Start the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Set frame size (optional)
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    import time
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Write the frame into the file 'output_video.avi'
        out.write(frame)

        # Display the resulting frame (optional)
        cv2.imshow('frame', frame)
        
        # Break the loop after 'duration' seconds
        if time.time() - start_time > duration:
            break
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release everything when the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Call the function to start recording
# This will record a 10-second video.
record_video(video_path='output_video.avi', duration=10)