#!/bin/bash

# Step 0: Optional - Setup a Python virtual environment (recommended)
echo "Setting up virtual environment..."
python3 -m venv yolov5env
source yolov5env/bin/activate

# Step 1: Check for Git and install if not exists
if ! command -v git &> /dev/null
then
    echo "Git could not be found. Attempting to install Git..."
    sudo apt-get update
    sudo apt-get install git
    echo "Git installed successfully."
else
    echo "Git is already installed."
fi

# Step 2: Clone YOLOv5 repository
echo "Cloning YOLOv5 repository..."
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Step 3: Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Step 4: Install PyTorch
# Note: Adjust the PyTorch installation command below based on your CUDA version
# or use the CPU version if CUDA isn't available
echo "Installing PyTorch..."
pip3 install torch torchvision torchaudio

pip3 install grpcio-tools

pip3 install Flask
pip3 install requests
sudo apt install protobuf-compiler

python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. object_tracking.proto

echo "YOLOv5 environment setup completed."