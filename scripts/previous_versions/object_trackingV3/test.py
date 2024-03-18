from modules.kalman_filter_module import KalmanFilter
from modules.kalman_utils import kalman_predict_and_update
def test_kalman():
    kf = KalmanFilter()
    detections = [(488.25543, 384.10526, 524.06842, 428.51779, 0.45080724, "67")]
    try:
        predicted_state = kalman_predict_and_update(kf, detections)
        print(f"Predicted state: {predicted_state}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_kalman()