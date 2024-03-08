import cv2
import numpy as np
import image_processing_pb2 as pb  # 假设生成的Python文件名为image_processing_pb2.py

def serialize_image(frame, format="JPEG"):
    """将OpenCV图像帧序列化为Protobuf格式"""
    success, encoded_image = cv2.imencode(f".{format.lower()}", frame)
    if not success:
        raise ValueError("图像编码失败")
    image = pb.Image(
        width=frame.shape[1],
        height=frame.shape[0],
        format=format,
        data=encoded_image.tobytes()
    )
    return image.SerializeToString()

def deserialize_image(serialized_image):
    """将序列化的Protobuf数据反序列化为OpenCV图像帧"""
    image_message = pb.Image()
    image_message.ParseFromString(serialized_image)
    frame = cv2.imdecode(
        np.frombuffer(image_message.data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    return frame

def process_video_stream():
    """从摄像头连续读取视频帧，序列化并反序列化，然后显示结果"""
    cap = cv2.VideoCapture(0)  # 使用摄像头

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        # 序列化图像帧
        serialized_frame = serialize_image(frame, "JPEG")

        # 反序列化图像帧
        deserialized_frame = deserialize_image(serialized_frame)

        # 在不同的窗口显示原始帧和反序列化的结果
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Deserialized Frame", deserialized_frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

process_video_stream()