#model_loader.py

import torch

def load_yolo_model(model_name='yolov5s', pretrained=True, device='cuda'):
    """
    Loads a YOLO model from the Ultralytics repository.
    
    Parameters:
    - model_name: str, the name of the YOLO model to load (e.g., 'yolov5s', 'yolov5m', etc.).
    - pretrained: bool, whether to load a pretrained model.
    - device: str, the device to load the model onto ('cpu' or 'cuda'). Default is 'cuda' to attempt GPU usage.
    
    Returns:
    - The loaded YOLO model.
    
    Notes:
    - This function allows for selecting different YOLO model variants and whether to use a pretrained model.
    - The model is loaded onto the specified device, with 'cuda' (GPU) as the default. If 'cuda' is not available, it will fall back to 'cpu'.
    - While this script does not implement a caching mechanism for the model itself (handled by torch.hub and setup scripts), it exemplifies how flexibility in resource usage can be achieved. For modules downloading large assets, consider implementing a caching check to see if the asset is available locally before downloading it, which is particularly beneficial for edge devices with limited connectivity or bandwidth.
    """
    
    # Check if CUDA is available and fall back to CPU if not or if 'cpu' is explicitly requested
    if not torch.cuda.is_available() or device == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda'
    
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained)
    
    # Move the model to the specified device
    model.to(device)
    
    return model