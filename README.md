# Traffic_Violence_Detection

This repository applies deep learning techniques to detect vehicles, identify traffic violations, and count the number of violations committed by motorcycles. The system integrates a vehicle detection model with DeepSORT for object tracking and violation counting.

Implementation Workflow:

• Train a license plate recognition model

• Train a vehicle detection model

• Train a motorcycle violation detection model

• Apply DeepSORT for tracking and violation counting on video data


## 🧠 Installation

Clone the repository:
   ```bash
   git clone https://github.com/locngocphan12/Traffic_Violence_Detection.git
   cd Traffic_Violence_Detection
   ```

## 📂 Dataset

We would like to thank the authors and sources of the following datasets used in this project:

- **[Vietnam License Plate Segment Datasets](https://www.kaggle.com/datasets/duydieunguyen/licenseplates)**  
  Used for training the license plate recognition model.

- **[Vehicle Detection 8 Classes | Object Detection](https://www.kaggle.com/datasets/sakshamjn/vehicle-detection-8-classes-object-detection), [Traffic vehicles Object Detection](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection), [VehicleDetection-YOLOv8](https://www.kaggle.com/datasets/alkanerturan/vehicledetection)**  
  Utilized YOLOv5 with pre-trained weights for vehicle detection.

- **[traffic violation dataset V.3](https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3)**  
  Applied for training the motorcycle traffic violation detection model.

## ▶️ How to run

### **Option 1: Run scripts locally**
1. Run `number_plate_detection.py`
2. Run `vehicle_detection.py`
3. Run `violence_detection.py`
4. Run `real_time_detection.py` using the saved model weights to generate the final output

### **Option 2: Run on Google Colab**
- Use the provided `.ipynb` notebook  
- Trained and tested on Google Colab with T4 GPU support

## ⚠️ Note

• Update all relevant **paths to data and trained model checkpoints** in the source code before running.
