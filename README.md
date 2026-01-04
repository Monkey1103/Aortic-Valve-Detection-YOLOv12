# Aortic Valve Detection based on YOLOv12

This repository contains the source code for the "Neural Network Final Project". The project focuses on detecting aortic valves in medical imaging using the YOLOv12x architecture.

## 📁 Project Structure
- `data_preprocess.py`: Handles data cleaning and generates empty label files for negative samples.
- `data_split_v2.py`: Implements the 1:1 negative sampling strategy and splits data into Train/Val sets.
- `train.py`: Training script using YOLOv12x (train from scratch).
- `test.py`: Inference script with "Minimum Area" post-processing logic.
- `visualize_data.py`: Tools for analyzing bounding box distributions.

## 🛠️ Environment Requirements
- Python 3.x
- Ultralytics (YOLOv12)
- OpenCV (cv2)
- NumPy
- Matplotlib

## 🚀 How to Run

### 1. Data Preprocessing
Organize your dataset and run the preprocessing script to handle labels:
```bash
python data_preprocess.py
```

### 2. Training
Train the model using the configuration file:
```bash
python train.py
```
Note: The model is trained from scratch for 500 epochs with a batch size of 6.

### 3. Inference (Testing)
Run the detection on test images. The script will automatically apply the post-processing algorithm (keeping only the smallest box):
```bash
python test.py
```

## 📊 Results
Model Architecture: YOLOv12x

Input Size: 640x640

Best Strategy: Random Negative Sampling (Model B)
