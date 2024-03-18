import torch
from model_loader import load_yolo_model

def test_load_yolo_model_on_cpu():
    print("Testing model loading on CPU...")
    model = load_yolo_model(device='cpu')
    assert model is not None, "Failed to load model on CPU"
    print("Model successfully loaded on CPU.")

def test_load_yolo_model_on_gpu():
    if torch.cuda.is_available():
        print("Testing model loading on GPU...")
        model = load_yolo_model(device='cuda')
        assert model is not None, "Failed to load model on GPU"
        print("Model successfully loaded on GPU.")
    else:
        print("GPU not available, skipping GPU test.")

if __name__ == "__main__":
    test_load_yolo_model_on_cpu()
    test_load_yolo_model_on_gpu()