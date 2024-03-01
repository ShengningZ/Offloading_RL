def filter_detections_by_mask(detections, fg_mask):
    filtered_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if fg_mask[center_y, center_x] > 0:
            filtered_detections.append(detection)
    return filtered_detections