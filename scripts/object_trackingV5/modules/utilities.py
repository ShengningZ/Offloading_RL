#utilities.py
def filter_detections_by_mask(detections, fg_mask):
    filtered_detections = []
    height, width = fg_mask.shape[:2]  # Get the dimensions of the fg_mask

    for detection in detections:
        if isinstance(detection, tuple) or isinstance(detection, list):
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = x1 - 5, y1 - 5, x2 + 5, y2 + 5  # 扩大检测框的大小
        else:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            x1, y1, x2, y2 = x1 - 5, y1 - 5, x2 + 5, y2 + 5  # 扩大检测框的大小
        
        points_to_check = [
            (int(x1), int(y1)),  # Top left
            (int(x2), int(y2)),  # Bottom right
            (int((x1 + x2) / 2), int((y1 + y2) / 2)),  # Center
        ]
        
        # Check if any of the points fall within the foreground areas, ensuring points are within bounds
        if any((0 <= x < width and 0 <= y < height and fg_mask[y, x] > 127) for x, y in points_to_check):
            filtered_detections.append(detection)
    return filtered_detections