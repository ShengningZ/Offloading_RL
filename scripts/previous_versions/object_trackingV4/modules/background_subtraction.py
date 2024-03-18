#background_subtraction.py

import cv2

def initialize_object_detector(history=100, varThreshold=40, detectShadows=True):
    """
    Initialize a background subtraction object detector using the MOG2 algorithm with customizable settings.

    Parameters:
    - history: The number of last frames that affect the background model.
    - varThreshold: Threshold on the squared Mahalanobis distance to decide whether it is well described
      by the background model. This parameter does not affect the background update.
    - detectShadows: Whether to detect and mark shadows in the output.

    Returns:
    - An initialized MOG2 background subtractor object.
    """
    return cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)

def apply_object_detector(detector, frame, threshold=254):
    print("Input frame shape:", frame.shape)
    print("Input frame dtype:", frame.dtype)
    cv2.imshow("Input Frame", frame)
    
    fg_mask = detector.apply(frame)
    print("Foreground mask shape:", fg_mask.shape)
    print("Foreground mask dtype:", fg_mask.dtype)
    cv2.imshow("Foreground Mask", fg_mask)
    
    _, fg_mask_thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
    print("Thresholded mask shape:", fg_mask_thresh.shape)
    print("Thresholded mask dtype:", fg_mask_thresh.dtype)
    cv2.imshow("Thresholded Mask", fg_mask_thresh)
    
    cv2.waitKey(1)  # Add a small delay to allow the windows to be displayed
    
    return fg_mask_thresh