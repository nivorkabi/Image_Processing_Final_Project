import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_dataset
import albumentations as A

# 1. Configuration & Reproducibility
random.seed(7)

# 2. Load the exact same 4 clean images
print("Loading dataset...")
ds = load_dataset("nateraw/ade20k-tiny", split="train")
idxs = random.sample(range(len(ds)), 4)
images = [ds[i]["image"] for i in idxs]

# 3. Define the distortion (Gaussian Noise) using Albumentations
# We apply a noticeable noise effect to see its clear impact on keypoints
transform = A.GaussNoise(var_limit=(10.0, 50.0), per_channel=False, p=1.0)

# 4. Initialize ORB detector
print("Initializing ORB detector...")
orb = cv2.ORB_create(nfeatures=500)

# 5. Apply noise and detect keypoints
print("Applying Gaussian Noise and running ORB...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, img_pil in zip(axes, images):
    img_np = np.array(img_pil)
    
    # Apply Albumentations distortion
    augmented = transform(image=img_np)
    distorted_img = augmented["image"]
    
    # Convert to grayscale for OpenCV ORB
    gray = cv2.cvtColor(distorted_img, cv2.COLOR_RGB2GRAY)
    
    # Detect keypoints and compute descriptors
    kp, des = orb.detectAndCompute(gray, None)
    
    # Draw keypoints on top of the distorted image
    img_with_kp = cv2.drawKeypoints(distorted_img, kp, None, color=(0, 255, 0), flags=0)
    
    ax.imshow(img_with_kp)
    ax.axis("off")

plt.tight_layout()
plt.show()