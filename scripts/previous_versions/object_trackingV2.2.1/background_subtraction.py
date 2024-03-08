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
    """
    Applies the initialized background subtractor to a frame to extract the foreground mask,
    and then applies thresholding to binarize the mask.

    Parameters:
    - detector: The background subtractor object.
    - frame: The input frame for foreground detection.
    - threshold: The threshold value for binarization of the foreground mask.

    Returns:
    - A binarized foreground mask.
    """
    # Apply the background subtractor to get the foreground mask
    fg_mask = detector.apply(frame)
    
    # Apply thresholding to binarize the mask
    _, fg_mask_thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)

    return fg_mask_thresh