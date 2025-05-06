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
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from glob import glob
from PIL import Image
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import kagglehub

# Download latest version
path = kagglehub.dataset_download("meliodassourav/traffic-violation-dataset-v3")

print("Path to dataset files:", path)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),          # Resize ảnh về 224x224
    transforms.ToTensor(),                  # Chuyển ảnh thành Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
])

label_mapping = {
    "helmet": 0,
    "no_helmet": 1,
    "overloading": 2
}

# 2. Modified image loading function with error handling
def load_images_from_folder(folder_path):
    images = []
    labels = []
    failed_images = []  # Track failed images

    all_images = []
    for label_name in label_mapping.keys():
        label_folder = os.path.join(folder_path, label_name)
        all_images.extend(glob(f"{label_folder}/*.jpg"))

    for img_path in tqdm(all_images, desc="Loading images"):
        # Get label from parent folder name
        label_name = os.path.basename(os.path.dirname(img_path))
        label_idx = label_mapping[label_name]

        # Try to read image
        img = cv2.imread(img_path)

        if img is not None:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                failed_images.append((img_path, f"Convert error: {str(e)}"))
        else:
            failed_images.append((img_path, "Read error: Image is None"))

    # Print summary
    print(f"\nSuccessfully loaded {len(images)} images")
    if failed_images:
        print(f"Failed to load {len(failed_images)} images:")
        for path, error in failed_images[:10]:  # Show first 10 failures
            print(f"- {path}: {error}")
        if len(failed_images) > 10:
            print(f"... and {len(failed_images) - 10} more")

    return images, labels

class HelmetViolationDataset:
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        # Convert numpy array to tensor if transform hasn't done it
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW format
            image = image / 255.0  # Normalize to [0,1]

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }


train_images, train_labels = load_images_from_folder("/kagglehub/datasets/meliodassourav/traffic-violation-dataset-v3/versions/1/Traffic Violations Analysis Dataset/Training data")
val_images, val_labels = load_images_from_folder("/kagglehub/datasets/meliodassourav/traffic-violation-dataset-v3/versions/1/Traffic Violations Analysis Dataset/validation data")
test_images, test_labels = load_images_from_folder("/kagglehub/datasets/meliodassourav/traffic-violation-dataset-v3/versions/1/Traffic Violations Analysis Dataset/Test data")

train_dataset = HelmetViolationDataset(train_images, train_labels, transform=transform)
val_dataset = HelmetViolationDataset(val_images, val_labels, transform=transform)
test_dataset = HelmetViolationDataset(test_images, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# main
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_item in pbar:
            images = batch_item['image'].to(device)
            labels = batch_item['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': running_loss/len(train_loader),
                            'acc': 100.*correct/total})

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_item in val_loader:
                images = batch_item['image'].to(device)
                labels = batch_item['label'].to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.*val_correct/val_total
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    return model

model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)

# Modify the classifier head
num_classes = 3
model.classifier[2] = nn.Linear(768, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)

model = train_model(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer,
        num_epochs = 20,
        device = DEVICE
    )

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_item in test_loader:
        images = batch_item['image'].to(DEVICE)
        labels = batch_item['label'].to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f'Test Accuracy: {100.*correct/total:.2f}%')

model = convnext_small(num_classes=3)
model_path = '/deepLearningProject/best_model.pth'
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

model.eval()
correct = 0
total = 0
y_true = []
y_pred = []
with torch.no_grad():
    for batch_item in test_loader:
        images = batch_item['image'].to(DEVICE)
        labels = batch_item['label'].to(DEVICE)
        outputs = model(images)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())


print(f'Test Accuracy: {100.*correct/total:.2f}%')

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['helmet', 'no_helmet', 'overloading'], yticklabels=['helmet', 'no_helmet', 'overloading'])
# Using trained model
class HelmetClassifier:
    def __init__(self, model_path='helmet_model.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize model
        self.model = convnext_small(num_classes=3)

        # Load saved weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # Class labels
        self.classes = ['helmet', 'no_helmet', 'overloading']

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            # If input is image path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image.to(self.device)

    def predict(self, image, return_confidence=False):
        """
        Predict class for single image
        Args:
            image: Can be image path (str) or numpy array
            return_confidence: If True, returns confidence scores
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.classes[predicted.item()]
            confidence = confidence.item()

        if return_confidence:
            return predicted_class, confidence
        return predicted_class

    def predict_batch(self, images):
        """Predict classes for a batch of images"""
        processed_images = torch.stack([self.preprocess_image(img).squeeze(0)
                                      for img in images])

        with torch.no_grad():
            outputs = self.model(processed_images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)

        return [self.classes[p.item()] for p in predicted]

# testing model
# Initialize model'
model_path = '/content/best_model.pth'
model_test = convnext_small(num_classes=3)

# Load saved weights
checkpoint = torch.load(model_path, map_location=DEVICE)
model_test.load_state_dict(checkpoint)
model_test = model.to(DEVICE)
model_test.eval()

img_path = '/content/sample_data/nguoi-nuoc-ngoai-thi-bang-lai.jpg'


def predict_image(image_path, model, device):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Map to class names
    classes = ['helmet', 'no_helmet', 'overloading']
    predicted_class = classes[predicted.item()]
    confidence = confidence.item()

    return predicted_class, confidence

predicted_class, confidence = predict_image(img_path, model_test, DEVICE)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}")

