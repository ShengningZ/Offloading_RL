import cv2
import numpy as np
from background_subtraction_pb2 import ForegroundMask

def serialize_foreground_mask(fg_mask, frame_shape):
    mask_data = cv2.imencode('.png', fg_mask)[1].tobytes()
    mask_message = ForegroundMask(mask_data=mask_data, width=frame_shape[1], height=frame_shape[0])
    return mask_message.SerializeToString()

def deserialize_foreground_mask(serialized_data):
    mask_message = ForegroundMask()
    mask_message.ParseFromString(serialized_data)
    mask_data = np.frombuffer(mask_message.mask_data, dtype=np.uint8)
    mask_image = cv2.imdecode(mask_data, cv2.IMREAD_UNCHANGED)
    return mask_image

def apply_object_detector_serialized(detector, frame, threshold=254):
    fg_mask = detector.apply(frame)
    _, fg_mask_thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
    return fg_mask_thresh

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fg_mask = apply_object_detector_serialized(background_subtractor, frame)
        serialized_mask = serialize_foreground_mask(fg_mask, frame.shape)
        deserialized_mask = deserialize_foreground_mask(serialized_mask)
        
        # 显示原始掩码和反序列化后的掩码进行比较
        cv2.imshow("Original Foreground Mask", fg_mask)
        cv2.imshow("Deserialized Foreground Mask", deserialized_mask)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/shengning/reinforcement rearning/scripts/object_trackingV2.2.1/output_video.avi"  # 更新为您的视频文件路径
    main(video_path)