# utilities.py
def filter_detections_by_mask(detections, fg_mask):
    filtered_detections = []
    height, width = fg_mask.shape[:2]  # Get the dimensions of the fg_mask

    for detection in detections:
        if len(detection) >= 4 and all(isinstance(coord, (int, float)) for coord in detection[:4]):
            x1, y1, x2, y2 = map(int, detection[:4])
            points_to_check = [
                (x1, y1),  # Top left
                (x2, y2),  # Bottom right
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),  # Center
                (x1, y2),  # Bottom left
                (x2, y1)   # Top right
            ]
            
            # Check if any of the points fall within the foreground areas, ensuring points are within bounds
            if any((0 <= x < width and 0 <= y < height and fg_mask[y, x] > 0) for x, y in points_to_check):
                filtered_detections.append(detection)
    
    return filtered_detections