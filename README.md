# Pattern Recognition-Final Project

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

### Step 3. Create YAML for YOLOv8
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

### Step 4. Train the model
```python
model = YOLO("yolov8s.pt") 

results = model.train(
    data=yaml_path,
    epochs=50,         
    imgsz=640,
    batch=16,
    patience=10,
    project="runs",
    name="yolov8",  
)

print("Training finished.")

```

### Step 5. Load best model and validate
```python
best_weights = "runs/yolov8/weights/best.pt"
best = YOLO(best_weights)
print("Model best loaded:", best_weights)

metrics = best.val(data=yaml_path)
print("Precision per class:", metrics.box.p)
print("Recall per class   :", metrics.box.r)
print("mAP50              :", metrics.box.map50)
print("mAP50-95           :", metrics.box.map)
```

### Step 6. Test on a random image from valid set
```python
import random

valid_img_dir = os.path.join(WORKING_ROOT, "valid", "images")
valid_images = [f for f in os.listdir(valid_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print("Number of valid images:", len(valid_images))

sample_img = random.choice(valid_images)
sample_path = os.path.join(valid_img_dir, sample_img)
print("Sample image:", sample_path)

res = best.predict(source=sample_path, conf=0.3)[0]
annotated = res.plot()  # BGR

annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(annotated_rgb)
plt.axis("off")
plt.title("Detections on valid sample")
plt.show()
```

### Step 7. Video inference with NMS + counting per frame
```python
import cv2
import torchvision.ops as tvops

video_source = "input/1.mp4"    
output_video = "output/1.mp4"

cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = best(frame, conf=0.35, iou=0.5, imgsz=640)[0]

    annotated = frame.copy()
    count_person = 0

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy  # [N, 4]
        conf = res.boxes.conf  # [N]
        cls  = res.boxes.cls   # [N]

        # keep only class 'person' (0)
        mask_person = (cls == 0)
        boxes = xyxy[mask_person]
        scores = conf[mask_person]

        if len(boxes) > 0:
            # secondary NMS to reduce duplicate boxes
            keep_idx = tvops.nms(boxes, scores, iou_threshold=0.5)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                score = scores[i].item()

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                ratio = bh / bw
                # filter extremely tall/skinny or flat boxes (poles, noise)
                if ratio < 0.8 or ratio > 4.5:
                    continue
                # filter very small boxes (noise)
                if bw < 15 or bh < 25:
                    continue

                count_person += 1
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(
                    annotated, f"person {score:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 2
                )

    cv2.putText(
        annotated, f"Person: {count_person}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (255,255,255), 2
    )

    out.write(annotated)
    frame_no += 1
    if frame_no % 20 == 0:
        print(f"Frame {frame_no}, Person: {count_person}")

cap.release()
out.release()
print("Finished. Saved video to:", output_video)
```
