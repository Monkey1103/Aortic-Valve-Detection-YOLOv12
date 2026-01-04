# Aortic Valve Detection based on YOLOv12

The project focuses on detecting aortic valves in medical imaging using the YOLOv12x architecture.

## 📁 Project Structure

### Python Scripts
- `data_preprocess.py`: Handles data cleaning and generates empty label files for negative samples.
- `data_split.py`: Initial data splitting script (random split).
- `data_split_v2.py`: **[Key Strategy]** Implements the 1:1 negative sampling strategy and splits data into Train/Val sets (Model B).
- `train.py`: Training script using YOLOv12x (train from scratch).
- `val.py`: Validation script to evaluate model performance on the validation set.
- `test.py`: Inference script with custom "Minimum Area" post-processing logic.
- `visualize_data.py`: Tools for analyzing bounding box distributions and calculating statistics.

### Configuration Files
- `yolo12x.yaml`: Configuration file defining the YOLOv12x model architecture.
- `aortic_valve_colab.yaml`: Dataset configuration file defining train/val paths and class names.

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
