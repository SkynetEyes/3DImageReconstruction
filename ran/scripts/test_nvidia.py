# test_fallDetec.py

import torch
from ultralytics import YOLO
import cv2

print("=== PyTorch & GPU Test ===")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

print("\n=== YOLO Test ===")
# Load a small pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # yolov8n = nano model, fast and small

# Create a dummy image (640x640 with 3 channels)
import numpy as np
dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

# Run detection
results = model(dummy_image)

# Print detected classes (should be empty on black image)
print("Detected classes:", results[0].boxes.cls if results[0].boxes else "None")

print("\nInstallation test complete!")

