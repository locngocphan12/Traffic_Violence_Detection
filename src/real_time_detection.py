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

# Load vehicle detection model
model = YOLO('/deepLearningProject/best_final.pt')
# Load two-wheelers violation
classifier = HelmetClassifier('/deepLearningProject/best_model.pth')
# Load number plate detection
license_plate_model = YOLO('/deepLearningProject/plate_recognition_model.pt')


# extracting frames from video
def extract_frames(video_path, output_frame_size=(640, 640)):
    """
    Trích xuất các khung hình từ một video.
    video_path: đường dẫn của 1 video

    return: frame_list: danh sách các khung hình được trích xuất
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    frame_list = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break
        # if frame_id == 150:
        #   break
        resized_frame = cv2.resize(frame, output_frame_size)
        frame_list.append(resized_frame)

        frame_id += 1

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Kích thước frame gốc: {frame_width}x{frame_height}")
    cap.release()
    return frame_list

from deep_sort_realtime.deepsort_tracker import DeepSort

# tracking id (if you want to check the efficiency of deepSORT
def deepSort_tracking_id(input_path, output_path):
    # Trích xuất frames từ video
    frames = extract_frames(input_path)
    frame_rate = 30  # Số frame trên giây
    tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.7)
    video_writer = None

    for i, frame in enumerate(frames):
        # Chuyển đổi màu sắc từ BGR sang RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Sử dụng mô hình YOLOv8 để phát hiện bounding boxes
        results = model(img)  # model là YOLOv8 đã được huấn luyện

        # Định dạng lại bounding boxes từ YOLOv8
        detected_boxes = []
        for pred in results[0].boxes:
            x1, y1, x2, y2 = pred.xyxy[0].cpu().numpy()
            conf = float(pred.conf.cpu().numpy())
            detected_boxes.append(([x1, y1, x2, y2], conf))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Cập nhật DeepSort tracker
        tracks = tracker.update_tracks(detected_boxes, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Vẽ bounding box và gán ID
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f'ID: {track_id}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        # Chuyển đổi lại từ RGB sang BGR để lưu video
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Khởi tạo VideoWriter nếu chưa có
        if video_writer is None:
            frame_height, frame_width = img_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

        # Ghi frame vào video
        video_writer.write(img_bgr)

    # Đóng VideoWriter
    if video_writer is not None:
        video_writer.release()

    print(f"Video đã được lưu tại {output_path}")

# countiing vehicle (this should be used for final result)
def deepSort_tracking(input_path, output_path):
  frames = extract_frames(input_path)
  frame_rate = 30  # Số frame trên giây
  frame_width = 1920
  frame_height = 1080
  tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.7)
  violated_ids = set()
  violation_buffer = {}  # Bộ đệm lưu trạng thái nhãn qua các frame
  buffer_size = 10  # Số lượng frame để xét bộ đệm
  violation_threshold = 7
  # processed_ids = set()
  video_writer = None

  for i, frame in enumerate(frames):
      motorbike_boxes = []
      # Chuyển đổi màu sắc từ BGR sang RGB
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


      results = model(img)

      for pred in results[0].boxes:
          x1, y1, x2, y2 = pred.xyxy[0].cpu().numpy()
          conf = float(pred.conf.cpu().numpy())
          class_id = int(pred.cls.cpu().numpy())
          label = model.names[class_id]
          if label == 'number_plate' or label == 'blur_number_plate':
            continue
          if label == 'motorcycle':
            motorbike_boxes.append([x1, y1, x2, y2, conf])
            x_min, y_min, x_max, y_max = map(int, pred.xyxy[0])
            cropped_frame = img[y_min:y_max, x_min:x_max]
            plate_results = license_plate_model(cropped_frame)

            for plate_pred in plate_results[0].boxes:
                  px1, py1, px2, py2 = map(int, plate_pred.xyxy[0])
                  cropped_plate = cropped_frame[py1:py2, px1:px2]

                  cv2.rectangle(img, (x_min + px1, y_min + py1), (x_min + px2, y_min + py2), (0, 0, 255), 2)

            predicted_class, confidence = classifier.predict(cropped_frame, return_confidence=True)
            if predicted_class == 'helmet':
              cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
              cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            text = f"{predicted_class} ({confidence:.2f})"
            text_x, text_y = x_min, y_min
            cv2.putText(
              img,
              text,
              (text_x, text_y),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (255, 0, 0),
              2
            )
          else:
            x_min, y_min, x_max, y_max = map(int, pred.xyxy[0])
            cropped_frame = img[y_min:y_max, x_min:x_max]
            plate_results = license_plate_model(cropped_frame)
            for plate_pred in plate_results[0].boxes:
                  px1, py1, px2, py2 = map(int, plate_pred.xyxy[0])
                  cropped_plate = cropped_frame[py1:py2, px1:px2]
                  cv2.rectangle(img, (x_min + px1, y_min + py1), (x_min + px2, y_min + py2), (0, 0, 255), 2)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

      # Định dạng lại motorbike_boxes
      motorbike_boxes_formatted = [
          ([x1, y1, x2, y2], conf) for x1, y1, x2, y2, conf in motorbike_boxes
      ]

      # Cập nhật tracks
      tracks = tracker.update_tracks(motorbike_boxes_formatted, frame=img)
      for track in tracks:
          if not track.is_confirmed():
              continue
          track_id = track.track_id
          ltrb = track.to_ltrb()
          x1, y1, x2, y2 = map(int, ltrb)
          crop_frame = img[y1:y2, x1:x2]
          predicted_class, confidence = classifier.predict(crop_frame, return_confidence=True)
          # if predicted_class != 'helmet' and confidence > 0.9:
          #   violated_ids.add(track_id)
          is_violation = predicted_class != 'helmet' and confidence > 0.9
          if track_id not in violation_buffer:
              violation_buffer[track_id] = []
          violation_buffer[track_id].append(is_violation)
          if len(violation_buffer[track_id]) > buffer_size:
              violation_buffer[track_id].pop(0)  # Xóa trạng thái cũ nhất

            # Kiểm tra nếu vượt ngưỡng vi phạm
          if sum(violation_buffer[track_id]) >= violation_threshold:
              violated_ids.add(track_id)
      frame_height, frame_width, _ = img.shape
      text_position = (frame_width - 250, 50)
      cv2.putText(img, f'Violation: {len(violated_ids)}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      #video saving
      if video_writer is None:
          frame_height, frame_width = img_bgr.shape[:2]
          fourcc = cv2.VideoWriter_fourcc(*"mp4v")
          video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
      video_writer.write(img_bgr)


  if video_writer is not None:
      video_writer.release()

  print(f"Video đã được lưu tại {output_path}")


# Counting violating vehicles - Modify your path
input_path = "/processed_video/sample_videos/"
output_path = "/processed_video/result_videos/"
deepSort_tracking(input_path, output_path)



