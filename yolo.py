import os
from pathlib import Path

import torch
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from ultralytics import YOLO

# Load model (will download if not available)
model = YOLO("yolov8m-seg.pt")

# Prepare images and dirs
image_dir = Path("data/images/wild/")
output_dir = Path("data/images/debug/annotated")
images = [image_dir / image for image in os.listdir(image_dir)]

# Run inference on batch
results = model(images, task="segment")

for i, result in enumerate(results):
    annotated_image = result.plot()
    output_path = f"{output_dir}/{result.path.split('/')[-1]}.jpg"
    # Save annotated images
    cv2.imwrite(output_path, annotated_image)
