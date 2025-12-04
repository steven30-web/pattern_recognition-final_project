# pattern_recognition-final_project

### Topic : Pedestrian and Crowd Detection <br>
### Objectives :
- Detect pedestrians and estimate crowd density in video surveillance footage.
- Understand object detection and multi-object tracking concepts.
### Datasets : human-crowd-dataset
### Network Model : YOLO("yolov8s.pt") <br><br>
### Step 0. Install and imports
```python
!pip install ultralytics kagglehub opencv-python matplotlib torchvision --quiet

import os
import shutil
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import torchvision.ops as tvops
```

### Step 1. Download the Datasets
```python
import kagglehub
BASE_PATH = kagglehub.dataset_download("hilongnguyn/human-crowd-dataset")
print("Dataset downloaded to:", BASE_PATH)
print("Splits:", os.listdir(BASE_PATH))
```

### Step 2. Copy train and valid to a working dataset
```python
WORKING_ROOT = "Datasets/human_crowd"

os.makedirs(WORKING_ROOT, exist_ok=True)

splits = ["train", "valid"]

for split in splits:
    src_images = os.path.join(BASE_PATH, split, "images")
    src_labels = os.path.join(BASE_PATH, split, "labels")

    dst_images = os.path.join(WORKING_ROOT, split, "images")
    dst_labels = os.path.join(WORKING_ROOT, split, "labels")

    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    print(f"Copying {split} set...")
    img_count = 0
    lbl_count = 0

    for fname in os.listdir(src_images):
        shutil.copy(os.path.join(src_images, fname), os.path.join(dst_images, fname))
        img_count += 1

    for fname in os.listdir(src_labels):
        shutil.copy(os.path.join(src_labels, fname), os.path.join(dst_labels, fname))
        lbl_count += 1

    print(f"  {split} images copied: {img_count}")
    print(f"  {split} labels copied: {lbl_count}")

print("\nDONE COPYING TRAIN + VALID")
print("WORKING_ROOT:", WORKING_ROOT)
print("Train images:", len(os.listdir(os.path.join(WORKING_ROOT, 'train', 'images'))))
print("Valid images:", len(os.listdir(os.path.join(WORKING_ROOT, 'valid', 'images'))))
```

### Step 3 : Create YAML for YOLOv8
```python
yaml_path = os.path.join(WORKING_ROOT, "human_crowd.yaml")

yaml_text = f"""
path: {WORKING_ROOT}
train: train/images
val: valid/images

names:
  0: person
"""

with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("Created YAML at:", yaml_path)
print(yaml_text)
```
