import random
import numpy as np
import cv2
from pathlib import Path
from datasets import load_dataset
import albumentations as A
from ultralytics import YOLO

# 1. Setup paths and reproducibility
random.seed(7)
np.random.seed(7)

print("=== Starting YOLOv8 Fine-Tuning Pipeline ===")

# Create directory structure required for YOLO training
work_dir = Path("yolo_dataset")
img_train_dir = work_dir / "images" / "train"
lbl_train_dir = work_dir / "labels" / "train"

img_train_dir.mkdir(parents=True, exist_ok=True)
lbl_train_dir.mkdir(parents=True, exist_ok=True)

# 2. Load the exact same 4 dataset images
ds = load_dataset("nateraw/ade20k-tiny", split="train")
idxs = random.sample(range(len(ds)), 4)
images = [ds[i]["image"] for i in idxs]

# 3. Load baseline model to generate Pseudo-Labels
base_model = YOLO("yolov8n.pt")
jpeg_distort = A.ImageCompression(quality_range=(5, 5), p=1.0)

def save_yolo_label(txt_path, boxes_xyxy, cls_ids, w, h):
    """Saves bounding boxes in YOLO normalized format (cls, cx, cy, bw, bh)"""
    lines = []
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, cls_ids):
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    txt_path.write_text("\n".join(lines))

# 4. Generate distorted images and corresponding pseudo-labels
print("Generating dataset and pseudo-labels from clean baseline...")
for i, img_pil in enumerate(images):
    img_clean = np.array(img_pil)
    h, w = img_clean.shape[:2]
    
    # Generate pseudo-labels using clean image inference
    res_clean = base_model.predict(img_clean, conf=0.35, iou=0.5, verbose=False)[0]
    if res_clean.boxes is None or len(res_clean.boxes) == 0:
        continue
        
    xyxy = res_clean.boxes.xyxy.cpu().numpy()
    cls_ids = res_clean.boxes.cls.cpu().numpy()
    
    # Apply JPEG compression distortion to the image
    distorted_img = jpeg_distort(image=img_clean)["image"]
    
    # Save distorted image and matching label file to the disk
    img_path = img_train_dir / f"im_{i}.jpg"
    lbl_path = lbl_train_dir / f"im_{i}.txt"
    
    cv2.imwrite(str(img_path), cv2.cvtColor(distorted_img, cv2.COLOR_RGB2BGR))
    save_yolo_label(lbl_path, xyxy, cls_ids, w, h)

# 5. Create data.yaml configuration file for YOLOv8
yaml_content = f"""
path: {work_dir.resolve()}
train: images/train
val: images/train

names:
"""
# Dynamically add class names from the base model configuration
for cid, cname in base_model.names.items():
    yaml_content += f"  {cid}: {cname}\n"

yaml_path = work_dir / "data.yaml"
yaml_path.write_text(yaml_content)

# 6. Initialize a fresh model and train it on the distorted images
print("Initializing fine-tuning on distorted images...")
ft_model = YOLO("yolov8n.pt")

# Train for 3 epochs with a small batch size as requested in the template
ft_model.train(
    data=str(yaml_path),
    imgsz=640,
    epochs=3,
    batch=2,
    device="cpu",
    verbose=True
)

print("\n=== Fine-Tuning Completed Successfully! ===")