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

path = kagglehub.dataset_download("sakshamjn/vehicle-detection-8-classes-object-detection")

print("Path to dataset files:", path)

def print_tree(directory, level=0):
    if not os.path.exists(directory):
        print(f"Path {directory} does not exist.")
        return

    files = os.listdir(directory)
    files.sort()

    num_files = sum(1 for f in files if os.path.isfile(os.path.join(directory, f)))

    for index, file in enumerate(files):
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            print("  " * level + f"|--- {file}")
            print_tree(path, level + 1)
        else:
            if num_files > 1 and index == 0:
                print("  " * level + f"|--- {file}")
            elif num_files > 1 and index == 1:
                print("  " * level + f"|--- ...")
                break
            else:
                print("  " * level + f"|--- {file}")

print_tree('/kagglehub/datasets/sakshamjn/vehicle-detection-8-classes-object-detection/versions/1/train')

def process_labels(label_folder, class_mapping):
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            file_path = os.path.join(label_folder, label_file)

            with open(file_path, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    if class_id in class_mapping:
                        parts[0] = str(class_mapping[class_id])
                    updated_lines.append(' '.join(parts))

            with open(file_path, 'w') as f:
                f.write('\n'.join(updated_lines))

# I use 3 different datasets for training vehicle detection

# Data 1
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sakshamjn/vehicle-detection-8-classes-object-detection")

print("Path to dataset files:", path)


source_images = '/kagglehub/datasets/sakshamjn/vehicle-detection-8-classes-object-detection/versions/1/train/images'
source_labels = '/kagglehub/datasets/sakshamjn/vehicle-detection-8-classes-object-detection/versions/1/train/labels'
destination_root = '/working/dataset'

os.makedirs(os.path.join(destination_root, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(destination_root, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(destination_root, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(destination_root, 'test/labels'), exist_ok=True)

images = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
labels = [f.replace('.jpg', '.txt') for f in images]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

for img, lbl in zip(train_images, train_labels):
    shutil.copy(os.path.join(source_images, img), os.path.join(destination_root, 'train/images', img))
    shutil.copy(os.path.join(source_labels, lbl), os.path.join(destination_root, 'train/labels', lbl))

for img, lbl in zip(test_images, test_labels):
    shutil.copy(os.path.join(source_images, img), os.path.join(destination_root, 'test/images', img))
    shutil.copy(os.path.join(source_labels, lbl), os.path.join(destination_root, 'test/labels', lbl))

# Data 2
import kagglehub

# Download latest version
path = kagglehub.dataset_download("saumyapatel/traffic-vehicles-object-detection")

print("Path to dataset files:", path)

class_mapping_1 = {0: 2, 1: 9, 2: 10, 3: 4, 4: 0, 5: 1, 6: 7}

folders_to_process = ['train', 'val']
root_folder = '/kagglehub/datasets/saumyapatel/traffic-vehicles-object-detection/versions/1/Traffic Dataset/labels'
# Duyệt qua từng thư mục con và xử lý
for folder in folders_to_process:
    label_folder = os.path.join(root_folder, folder)
    if os.path.exists(label_folder):
        process_labels(label_folder, class_mapping_1)
    else:
        print(f"Folder {label_folder} không tồn tại!")

source_images_train = '/kagglehub/datasets/saumyapatel/traffic-vehicles-object-detection/versions/1/Traffic Dataset/images/train'
source_images_test = '/kagglehub/datasets/saumyapatel/traffic-vehicles-object-detection/versions/1/Traffic Dataset/images/val'
source_labels_train = '/kagglehub/datasets/saumyapatel/traffic-vehicles-object-detection/versions/1/Traffic Dataset/labels/train'
source_labels_test = '/kagglehub/datasets/saumyapatel/traffic-vehicles-object-detection/versions/1/Traffic Dataset/labels/val'
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
# Train set
train_images = [f for f in os.listdir(source_images_train) if f.endswith(valid_extensions)]
train_labels = [os.path.splitext(f)[0] + '.txt' for f in train_images]
for img, lbl in zip(train_images, train_labels):
    shutil.copy(os.path.join(source_images_train, img), os.path.join(destination_root, 'train/images', img))
    shutil.copy(os.path.join(source_labels_train, lbl), os.path.join(destination_root, 'train/labels', lbl))
# Test set
test_images = [f for f in os.listdir(source_images_test) if f.endswith(valid_extensions)]
test_labels = [os.path.splitext(f)[0] + '.txt' for f in test_images]
for img, lbl in zip(test_images, test_labels):
    shutil.copy(os.path.join(source_images_test, img), os.path.join(destination_root, 'test/images', img))
    shutil.copy(os.path.join(source_labels_test, lbl), os.path.join(destination_root, 'test/labels', lbl))



# Data 3
import kagglehub

# Download latest version
path = kagglehub.dataset_download("alkanerturan/vehicledetection")

print("Path to dataset files:", path)

class_mapping_2 = {0: 8, 3: 4, 4: 7}
folders_to_process = ['train/labels', 'valid/labels']
root_folder = '/root/.cache/kagglehub/datasets/alkanerturan/vehicledetection/versions/3/VehiclesDetectionDataset'
# Duyệt qua từng thư mục con và xử lý
for folder in folders_to_process:
    label_folder = os.path.join(root_folder, folder)
    if os.path.exists(label_folder):
        process_labels(label_folder, class_mapping_2)
    else:
        print(f"Folder {label_folder} không tồn tại!")

source_images_train = '/kagglehub/datasets/alkanerturan/vehicledetection/versions/3/VehiclesDetectionDataset/train/images'
source_images_test = '/kagglehub/datasets/alkanerturan/vehicledetection/versions/3/VehiclesDetectionDataset/valid/images'
source_labels_train = '/kagglehub/datasets/alkanerturan/vehicledetection/versions/3/VehiclesDetectionDataset/train/labels'
source_labels_test = '/kagglehub/datasets/alkanerturan/vehicledetection/versions/3/VehiclesDetectionDataset/valid/labels'
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Train set
train_images = [f for f in os.listdir(source_images_train) if f.endswith(valid_extensions)]
train_labels = [os.path.splitext(f)[0] + '.txt' for f in train_images]
for img, lbl in zip(train_images, train_labels):
    shutil.copy(os.path.join(source_images_train, img), os.path.join(destination_root, 'train/images', img))
    shutil.copy(os.path.join(source_labels_train, lbl), os.path.join(destination_root, 'train/labels', lbl))

# Test set
test_images = [f for f in os.listdir(source_images_test) if f.endswith(valid_extensions)]
test_labels = [os.path.splitext(f)[0] + '.txt' for f in test_images]
for img, lbl in zip(test_images, test_labels):
    shutil.copy(os.path.join(source_images_test, img), os.path.join(destination_root, 'test/images', img))
    shutil.copy(os.path.join(source_labels_test, lbl), os.path.join(destination_root, 'test/labels', lbl))

# Format data.yaml

data_yaml = {
    'train': '/working/dataset/train/images',
    'val': '/working/dataset/test/images',
    'names': ['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck', 'ambulance', 'number_plate', 'blur_number_plate']
}

with open('/working/dataset/data.yaml', 'w') as file:
    yaml.dump(data_yaml, file, default_flow_style=False)

model = YOLO('yolov8m.pt')

model.train(data='/working/dataset/data.yaml',
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

# testing
# Insert fine-tuned model path
model = YOLO('/content/working/dataset/runs/detect/train/weights/best.pt')
test_image_dir = '/working/dataset/test/images'
test_images = os.listdir(test_image_dir)
test_images = [img for img in test_images if img.endswith('.jpg')]
sample_images = random.sample(test_images, 5)

fig, axes = plt.subplots(len(sample_images), 1, figsize=(10, len(sample_images) * 5))

for i, image_name in enumerate(sample_images):
    image_path = os.path.join(test_image_dir, image_name)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(image_path)
    axes[i].imshow(img_rgb)
    axes[i].axis('off')

    for pred in results[0].boxes:
        x1, y1, x2, y2 = pred.xyxy[0].cpu().numpy()
        conf = float(pred.conf.cpu().numpy())
        class_id = int(pred.cls.cpu().numpy())
        label = model.names[class_id]

        axes[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
        axes[i].text(x1, y1, f'{label}: {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

plt.tight_layout()
plt.show()