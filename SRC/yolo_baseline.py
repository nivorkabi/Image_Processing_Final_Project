import random
import matplotlib.pyplot as plt
from datasets import load_dataset
from ultralytics import YOLO

# 1. Configuration & Reproducibility
random.seed(7)

# 2. Load the exact same 4 clean images from the dataset
print("Loading dataset...")
ds = load_dataset("nateraw/ade20k-tiny", split="train")
idxs = random.sample(range(len(ds)), 4)
images = [ds[i]["image"] for i in idxs]

# 3. Load Pretrained YOLOv8 Model (Nano version - lightweight and fast)
print("Loading pretrained YOLOv8 model...")
model = YOLO("yolov8n.pt") 

# 4. Run Object Detection Inference on Clean Images
print("Running YOLOv8 inference...")
results = model(images)

# 5. Plot the Detection Results in a Grid
print("Plotting object detection results...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, res in zip(axes, results):
    # res.plot() returns a BGR numpy array with drawn bounding boxes and labels
    # We convert it to RGB using [:, :, ::-1] so Matplotlib displays colors correctly
    annotated_img = res.plot()[:, :, ::-1]
    
    ax.imshow(annotated_img)
    ax.axis("off")

plt.tight_layout()
plt.show()