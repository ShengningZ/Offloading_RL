
import numpy as np
import cv2
import torch
import time
import os

from modules.model_loader import load_yolo_model
from modules.object_detection import detect_objects_yolo
from modules.kalman_filter_module import KalmanFilter
from modules.background_subtraction import initialize_object_detector, apply_object_detector
from modules.visualization import visualize_mask, visualize_detections, visualize_kalman_prediction,visualize_filtered_detections
from modules.utilities import filter_detections_by_mask
from modules.kalman_utils import kalman_predict_and_update, update_kalman_with_contours, adjust_kalman_parameters

def should_offload_operation():
    # Placeholder for logic to decide whether to offload processing
    # Could be based on network conditions, device load, etc.
    return False

def convert_detections_to_tensor(detections):
    return [torch.tensor([d[0], d[1], d[2], d[3], d[4], 0]) for d in detections]

def draw_detections_and_mask(frame, detections, fg_mask):
    fg_mask_uint8 = cv2.bitwise_not(fg_mask)  # 反转掩码以获取正确的颜色
    mask_vis = cv2.bitwise_and(frame, frame, mask=fg_mask_uint8.astype(np.uint8))
    detections_vis = frame.copy()
    visualize_detections(detections_vis, detections, show_confidence=True)
    combined = cv2.addWeighted(mask_vis, 0.5, detections_vis, 0.5, 0)
    cv2.imshow("Detections and Mask", combined)


def main():
    # 初始化 CUDA (如果可用)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 初始化组件
    yolo_model = load_yolo_model(device=device)
    kf = KalmanFilter()
    object_detector = initialize_object_detector()

    cap = cv2.VideoCapture(0)  # 假设使用网络摄像头

    last_measurement = None
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 执行背景减除
            fg_mask_thresh = apply_object_detector(object_detector, frame)
            visualize_mask(fg_mask_thresh)

            # 执行目标检测
            if should_offload_operation():
                # 卸载目标检测操作 (此处为占位符,实际需要实现卸载功能)
                detections = offload_object_detection(frame)
            else:
                # 本地执行目标检测
                detections = detect_objects_yolo(frame, yolo_model, device=device)
                detections = [torch.tensor([d[0], d[1], d[2], d[3], d[4], int(d[5])], device=device) for d in detections.tolist()]
                detections = [d.cpu() for d in detections]  # 将检测结果移动到CPU

            # 执行检测结果过滤
            filtered_detections = filter_detections_by_mask(detections, fg_mask_thresh)
            filtered_detections = convert_detections_to_tensor(filtered_detections)

            elapsed_time = time.time() - start_time


            if elapsed_time >= 5:  # 在5秒后捕获图像
                # 保存原始帧
                cv2.imwrite("original_frame.png", frame)

                # 保存过滤后的检测结果
                filtered_detections_vis = frame.copy()
                visualize_filtered_detections(filtered_detections_vis, filtered_detections, fg_mask_thresh)
                cv2.imwrite("filtered_detections.png", filtered_detections_vis)

                # 保存未过滤的检测结果
                detections_vis = frame.copy()
                visualize_detections(detections_vis, detections, show_confidence=True)
                cv2.imwrite("detections.png", detections_vis)

                # 保存卡尔曼滤波预测结果
                prediction_frame = frame.copy()
                visualize_kalman_prediction(prediction_frame, predicted_state, kf.kf.statePost[2:4])
                cv2.imwrite("prediction.png", prediction_frame)

                # 保存前景掩码
                cv2.imwrite("foreground_mask.png", fg_mask_thresh)

                # 保存前景掩码和检测结果合并图像
                
                mask_and_detections = frame.copy()
                detections_vis_copy = mask_and_detections.copy()
                visualize_detections(detections_vis_copy, filtered_detections, show_confidence=True)
                fg_mask_uint8 = cv2.bitwise_not(fg_mask_thresh)
                mask_vis = cv2.bitwise_and(mask_and_detections, mask_and_detections, mask=fg_mask_uint8.astype(np.uint8))
                combined = cv2.addWeighted(mask_vis, 0.5, detections_vis_copy, 0.5, 0)
                cv2.imwrite("mask_and_detections.png", combined)

                break
                break

            # 基于检测结果更新卡尔曼滤波器
            predicted_state = kalman_predict_and_update(kf, filtered_detections)

            # 基于轮廓更新卡尔曼滤波器
            update_kalman_with_contours(kf, frame, object_detector)

            # 根据目标运动速度调整卡尔曼滤波器参数
            current_measurement = kf.kf.statePost[:2]
            if last_measurement is not None:
                adjust_kalman_parameters(kf, last_measurement, current_measurement)
            last_measurement = current_measurement

            # 可视化检测结果
            detections_frame = frame.copy()
            visualize_detections(detections_frame, filtered_detections, show_confidence=True)
            cv2.imshow("Detections", detections_frame)
            cv2.waitKey(1)  # 添加一个小延迟以显示窗口

            # 可视化卡尔曼滤波预测结果
            visualize_kalman_prediction(frame, predicted_state, kf.kf.statePost[2:4])

            # 实时显示前景掩码和检测结果合并图像
            draw_detections_and_mask(frame, filtered_detections, fg_mask_thresh)

            if cv2.waitKey(30) & 0xFF == 27:
                break

    except Exception as e:
        print(f"发生异常: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()