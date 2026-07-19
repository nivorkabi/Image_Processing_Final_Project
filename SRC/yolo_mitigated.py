import random
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from ultralytics import YOLO

# 1. Configuration & Reproducibility
random.seed(7)

# 2. Load the exact same 4 clean images
print("Loading dataset...")
ds = load_dataset("nateraw/ade20k-tiny", split="train")
idxs = random.sample(range(len(ds)), 4)
images = [ds[i]["image"] for i in idxs]

# 3. Function to apply severe JPEG compression artifact
def apply_jpeg_distortion(img_pil, quality=5):
    """Saves the image in memory with very low JPEG quality to create artifacts."""
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# 4. Function to apply Bilateral Filtering for restoration
def apply_bilateral_filter(img_pil):
    """Converts image to numpy, applies Bilateral Filter to smooth compression artifacts while preserving edges."""
    img_np = np.array(img_pil)
    # d=9 (pixel neighborhood), sigmaColor=75, sigmaSpace=75
    restored_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(restored_np)

print("Processing images (Distortion -> Bilateral Restoration)...")
distorted_images = [apply_jpeg_distortion(img) for img in images]
restored_images = [apply_bilateral_filter(img) for img in distorted_images]

# 5. Load Pretrained YOLOv8 Model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt") 

# 6. Run Inference on Restored Images
print("Running YOLOv8 inference on restored images...")
results = model(restored_images)

# 7. Plot the Restored Detection Results
print("Plotting restored results...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, res in zip(axes, results):
    annotated_img = res.plot()[:, :, ::-1] # Convert BGR to RGB
    ax.imshow(annotated_img)
    ax.axis("off")

plt.tight_layout()
plt.show()