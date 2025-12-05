# Pattern Recognition-Final Project

### Topic : Pedestrian and Crowd Detection </br>
### Objectives :
- Detect pedestrians and estimate crowd density in video surveillance footage.
- Understand object detection and multi-object tracking concepts.
### Datasets : human-crowd-dataset
### Network Model : YOLO("yolov8s.pt") </br></br>
### Experiments
#### Step 0. Install and imports </br></br>
#### Step 1. Download the Datasets
Human Crowd </br>
https://www.kaggle.com/datasets/hilongnguyn/human-crowd-dataset </br></br>
#### Step 2. Copy Train and Valid Dataset Into a Writable Working Directory
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
</br>

#### Step 3. Generate YOLO-Formatted Pseudo-Labels for Train and Valid Sets
```python
import os
import shutil
from ultralytics import YOLO
import cv2
import glob
from tqdm import tqdm
import numpy as np

auto_label_model = YOLO("yolov8s.pt")

for split in ["train", "valid"]:
    images_root = os.path.join(WORKING_ROOT, split, "images")
    labels_root = os.path.join(WORKING_ROOT, split, "labels")

    print(f"\n=== Generating pseudo-labels for {split} ===")
    print("Images root:", images_root)
    print("Labels root (will be refreshed):", labels_root)

    if os.path.exists(labels_root):
        shutil.rmtree(labels_root)
    os.makedirs(labels_root, exist_ok=True)
    print("Labels root refreshed:", labels_root)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(images_root, ext)))

    for img_path in tqdm(image_paths):
        res = auto_label_model(img_path, conf=0.3, iou=0.5, classes=[0])[0]

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape

        labels = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 1 or bh <= 1:
                    continue

                xc = (x1 + x2) / 2.0 / w
                yc = (y1 + y2) / 2.0 / h
                nw = bw / w
                nh = bh / h

                labels.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_root, base + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

    print(f"Pseudo-label {split} in:", labels_root)
    print("Labels File:", os.listdir(labels_root)[:5])
```
</br>

#### Step 4. Buat file human_crowd.yaml untuk YOLOv8 </br></br>
#### Step 5. Training YOLOv8 pada dataset Human Crowd
```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt") 

results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    patience=20,
    project="runs",
    name="yolov8",
)
```

</br>

#### Step 6. Load best model and validate


#### Step 7. Test on a random image from valid set
```python
import random
import matplotlib.pyplot as plt
%matplotlib inline

img_files = [f for f in os.listdir(images_root) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
sample_img = random.choice(img_files)
sample_path = os.path.join(images_root, sample_img)
print("Sample image:", sample_path)

res = best.predict(source=sample_path, conf=0.3)[0]
annotated = res.plot() 

annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(annotated_rgb)
plt.axis("off")
plt.title("People Detection - Human Crowd (fine-tuned)")
plt.show()
```

#### Step 8. Upload a Video And the Model Detect Human Crowd
#### Video Input </br>
https://github.com/steven30-web/pattern_recognition-final_project/blob/main/Input/1.mp4 

#### Video Output 

