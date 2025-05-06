import torch.nn as nn
import torch
import json
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
from deep_sort_realtime.deepsort_tracker import DeepSort
import pytesseract

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import kagglehub

# Download latest version
path = kagglehub.dataset_download("duydieunguyen/licenseplates")

print("Path to dataset files:", path)


# Modify specific dataset paths
# check path
print(os.listdir('/kagglehub/datasets/duydieunguyen/licenseplates/versions/1/'))

data_yaml = {
    'train': '/kagglehub/datasets/duydieunguyen/licenseplates/versions/1/images/train',
    'val': '/kagglehub/datasets/duydieunguyen/licenseplates/versions/1/images/val',
    'names': ['BSD', 'BSV']
}

with open('/kagglehub/datasets/duydieunguyen/licenseplates/versions/1/data.yaml', 'w') as file:
    yaml.dump(data_yaml, file, default_flow_style=False)

model = YOLO('yolov8m.pt')

model.train(data='/kagglehub/datasets/duydieunguyen/licenseplates/versions/1/data.yaml',
    epochs=20,
    batch=16,
    imgsz=640,
    mosaic=1.0,                        # Mức độ áp dụng Mosaic augmentation
    hsv_h=0.015,                       # Điều chỉnh Hue
    hsv_s=0.7,                         # Điều chỉnh Saturation
    hsv_v=0.4,                         # Điều chỉnh Brightness
    degrees=10.0,                      # Góc xoay
    translate=0.1,                     # Tịnh tiến
    scale=0.5,                         # Phóng to/thu nhỏ
    shear=0.1
    )

