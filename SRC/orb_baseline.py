import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_dataset

# 1. Configuration & Reproducibility
random.seed(7)

# 2. Load the exact same 4 clean images
print("Loading dataset...")
ds = load_dataset("nateraw/ade20k-tiny", split="train")
idxs = random.sample(range(len(ds)), 4)
images = [ds[i]["image"] for i in idxs]

# 3. Initialize ORB detector
print("Initializing ORB detector...")
orb = cv2.ORB_create(nfeatures=500)

# 4. Detect and draw keypoints
print("Detecting keypoints...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, img_pil in zip(axes, images):
    # Convert PIL to grayscale numpy array for OpenCV ORB
    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Detect keypoints and compute descriptors
    kp, des = orb.detectAndCompute(gray, None)
    
    # Draw keypoints on top of the original image
    img_with_kp = cv2.drawKeypoints(img_np, kp, None, color=(0, 255, 0), flags=0)
    
    ax.imshow(img_with_kp)
    ax.axis("off")

plt.tight_layout()
plt.show()