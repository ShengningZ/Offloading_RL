import sys
import os
import time

# 获取当前脚本的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加 communication 目录到 sys.path
communication_dir = os.path.join(current_dir, 'communication')
sys.path.append(communication_dir)
import numpy as np
import cv2
import grpc
import torch
import communication.project_data_pb2 as project_data_pb2
import communication.project_data_pb2_grpc as project_data_pb2_grpc
from modules.background_subtraction import initialize_object_detector, apply_object_detector
from modules.object_detection import detect_objects_yolo
from modules.utilities import filter_detections_by_mask
from modules.kalman_filter_module import KalmanFilter
from modules.model_loader import load_yolo_model
from modules.visualization import visualize_mask, visualize_detections, visualize_kalman_prediction
from mainwithremote import convert_detections_to_tensor,remote_background_subtraction,local_background_subtraction,remote_object_detection,local_object_detection,remote_filtering,local_filtering,remote_kalman_filter,local_kalman_filter

class QLearning:
    def __init__(self, state_space, action_space, learning_rate, discount_factor, exploration_rate, convergence_threshold):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((len(state_space), len(action_space)))
        self.convergence_threshold = convergence_threshold
        self.max_q_change = float('inf')

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[state, :])
        return action

    def update_q_value(self, state, action, reward, next_state):
        try:
            old_q = self.q_table[state, action]
            predict = old_q
            target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
            self.q_table[state, action] += self.learning_rate * (target - predict)
            new_q = self.q_table[state, action]
            self.max_q_change = max(self.max_q_change, abs(new_q - old_q))
        except IndexError:
            print(f"Invalid indices: state={state}, action={action}")

    def simulate_action(self, state, action_index, frame, object_detector, yolo_model, bg_subtraction_stub, object_detection_stub, filtering_stub, kalman_filter_stub):
        offload_config = self.get_offload_config(action_index)
        latency = self.run_with_offload_config(frame, offload_config, object_detector, yolo_model, bg_subtraction_stub, object_detection_stub, filtering_stub, kalman_filter_stub)
        new_state = state  # assume new state doesn't change
        return new_state, latency

    def get_offload_config(self, action_index):
        action = self.action_space[action_index]
        offload_config = {
            'background_subtraction': action[0],
            'object_detection': action[1],
            'filtering': action[2],
            'kalman_filter': action[3]
        }
        return offload_config

    def train(self, frame, object_detector, yolo_model, bg_subtraction_stub, object_detection_stub, filtering_stub, kalman_filter_stub):
        min_latency = float('inf')
        best_offload_configs = [None, None, None]  # 存储前三个最佳卸载配置
        while self.max_q_change > self.convergence_threshold:
            self.max_q_change = 0
            for action_index, action in enumerate(self.action_space):
                state = self.state_space[0]  # Assuming the initial state is 0
                total_latency = 0
                for test_iter in range(100):  # stop condition: test each action 100 times
                    new_state, latency = self.simulate_action(state, action_index, frame, object_detector, yolo_model, bg_subtraction_stub, object_detection_stub, filtering_stub, kalman_filter_stub)
                    reward = -latency  # use actual latency as negative reward
                    self.update_q_value(state, action_index, reward, new_state)
                    total_latency += latency
                    state = new_state
                total_latency /= 100  # calculate average latency for the action
                if total_latency < min_latency:
                    min_latency = total_latency
                    best_offload_configs[0] = action
                    best_offload_configs[1] = best_offload_configs[0]
                    best_offload_configs[2] = best_offload_configs[1]
                elif total_latency < min_latency + self.convergence_threshold:
                    best_offload_configs[1] = action
                    best_offload_configs[2] = best_offload_configs[1]
                elif total_latency < min_latency + 2 * self.convergence_threshold:
                    best_offload_configs[2] = action
        return best_offload_configs


    def run_with_offload_config(self, frame, offload_config, object_detector, yolo_model, bg_subtraction_stub, object_detection_stub, filtering_stub, kalman_filter_stub):
        start_time = time.time()
        kf = KalmanFilter()

        if offload_config['background_subtraction']:
            print("offloading bc")
            fg_mask_thresh = remote_background_subtraction(bg_subtraction_stub, frame)
        else:
            print("locally processing bc")
            fg_mask_thresh = local_background_subtraction(object_detector, frame)

        visualize_mask(fg_mask_thresh)

        if offload_config['object_detection']:
            print("offloading od")
            detections = remote_object_detection(object_detection_stub, frame)
        else:
            print("locally processing od")
            detections = local_object_detection(frame, yolo_model)
            detections = detections.cpu().numpy().tolist()
            detections = [torch.tensor(d) for d in detections]  # Convert to list of PyTorch tensors

        if offload_config['filtering']:
            print("offloading filteting")
            filtered_detections = remote_filtering(filtering_stub, detections, fg_mask_thresh)
        else:
            print("locally filtering")
            filtered_detections = local_filtering(detections, fg_mask_thresh)

        print(f"Type of filtered_detections: {type(filtered_detections)}")
        print(f"Filtered detections: {filtered_detections}")
        
        if offload_config['kalman_filter'] and len(filtered_detections) > 0:
            print("offloading kalman")
            # Convert filtered detections to protobuf message format
            detections_proto = [project_data_pb2.Detection(x1=d[0].item(), y1=d[1].item(), x2=d[2].item(), y2=d[3].item(), confidence=d[4].item(), label=str(int(d[5].item()))) for d in filtered_detections]

            # Create KalmanFilterRequest message
            kalman_request = project_data_pb2.KalmanFilterRequest(detection_result=project_data_pb2.DetectionResult(detections=detections_proto))

            # Call remote Kalman filter
            predicted_state, state_post = remote_kalman_filter(kalman_filter_stub, kalman_request)
        elif len(filtered_detections) > 0:
            print("local kalman")
            predicted_state = local_kalman_filter(kf, filtered_detections)
            state_post = kf.kf.statePost
        else:
            predicted_state = None
            state_post = None

        print("Predicted state:", predicted_state)
        if state_post is not None:
            velocity = state_post[2:4]
            print(f"Velocity: vx={velocity[0]}, vy={velocity[1]}")

        # Visualize detections
        visualize_detections(frame, filtered_detections)

        if predicted_state is not None:
            visualize_kalman_prediction(frame, predicted_state, velocity)

        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time

    def main(self, cap, object_detector, yolo_model, bg_subtraction_stub, object_detection_stub, filtering_stub, kalman_filter_stub):
        training_finished = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not training_finished:
                print("Training Q-learning model")
                best_offload_configs = self.train(frame, object_detector, yolo_model, bg_subtraction_stub, object_detection_stub, filtering_stub, kalman_filter_stub)
                training_finished = True
                print(f"Best offload config: {best_offload_configs[0]}")
                print(f"Second best offload config: {best_offload_configs[1]}")
                print(f"Third best offload config: {best_offload_configs[2]}")
                print("Training completed. Press 'q' to quit.")
            else:
                best_offload_config = best_offload_configs[0]
                print(f"Using best offload config: {best_offload_config}")

                if best_offload_config[0]:  # Background subtraction
                    fg_mask_thresh = remote_background_subtraction(bg_subtraction_stub, frame)
                else:
                    fg_mask_thresh = local_background_subtraction(object_detector, frame)

                if best_offload_config[1]:  # Object detection
                    detections = remote_object_detection(object_detection_stub, frame)
                else:
                    detections = local_object_detection(frame, yolo_model)
                    detections = detections.cpu().numpy().tolist()
                    detections = [torch.tensor(d) for d in detections]  # Convert to list of PyTorch tensors

                if best_offload_config[2]:  # Filtering
                    filtered_detections = remote_filtering(filtering_stub, detections, fg_mask_thresh)
                else:
                    filtered_detections = local_filtering(detections, fg_mask_thresh)

                if best_offload_config[3] and len(filtered_detections) > 0:  # Kalman filter
                    predicted_state, state_post = remote_kalman_filter(kalman_filter_stub, filtered_detections)
                elif len(filtered_detections) > 0:
                    kf = KalmanFilter()
                    predicted_state = local_kalman_filter(kf, filtered_detections)
                    state_post = kf.kf.statePost
                else:
                    predicted_state = None
                    state_post = None

                visualize_mask(fg_mask_thresh)
                visualize_detections(frame, filtered_detections)
                if predicted_state is not None:
                    visualize_kalman_prediction(frame, predicted_state, state_post[2:4])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize components
    yolo_model = load_yolo_model()
    cap = cv2.VideoCapture(0)  # Assume using the webcam
    object_detector = initialize_object_detector()

    # Connect to gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    bg_subtraction_stub = project_data_pb2_grpc.BackgroundSubtractionServiceStub(channel)
    object_detection_stub = project_data_pb2_grpc.ObjectDetectionServiceStub(channel)
    filtering_stub = project_data_pb2_grpc.FilteringServiceStub(channel)
    kalman_filter_stub = project_data_pb2_grpc.KalmanFilterServiceStub(channel)

    # Q-learning parameters
    learning_rate = 0.1
    discount_factor = 0.9
    exploration_rate = 0.1
    convergence_threshold = 0.01

    # Define state space and action space
    state_space = [0, 1]
    action_space = [(a, b, c, d) for a in [False, True] for b in [False, True] for c in [False, True] for d in [False, True]]

    # Initialize Q-learning agent
    q_agent = QLearning(state_space, action_space, learning_rate, discount_factor, exploration_rate, convergence_threshold)

    # Run the main program
    q_agent.main(cap, object_detector, yolo_model, bg_subtraction_stub, object_detection_stub, filtering_stub, kalman_filter_stub)