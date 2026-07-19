import random
import io
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

# Apply the distortion to all 4 images
print("Applying severe JPEG compression (Quality = 5)...")
distorted_images = [apply_jpeg_distortion(img) for img in images]

# 4. Load Pretrained YOLOv8 Model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt") 

# 5. Run Inference on Distorted Images
print("Running YOLOv8 inference on distorted images...")
results = model(distorted_images)

# 6. Plot the Distorted Detection Results
print("Plotting distorted results...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, res in zip(axes, results):
    annotated_img = res.plot()[:, :, ::-1] # Convert BGR to RGB
    ax.imshow(annotated_img)
    ax.axis("off")

plt.tight_layout()
plt.show()