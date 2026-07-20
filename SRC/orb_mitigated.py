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
transform = A.GaussNoise(var_limit=(10.0, 50.0), per_channel=False, p=1.0)

# 4. Initialize ORB detector
print("Initializing ORB detector...")
orb = cv2.ORB_create(nfeatures=500)

# 5. Process: Add Noise -> Apply Fast NLM -> Run ORB
print("Applying Gaussian Noise, Fast NLM restoration, and running ORB...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, img_pil in zip(axes, images):
    img_np = np.array(img_pil)
    
    # Apply distortion
    augmented = transform(image=img_np)
    distorted_img = augmented["image"]
    
    # Apply Fast Non-Local Means Denoising for Color Images
    # h=15 controls the filter strength (higher means more denoising)  
    # We got bad results for h=15, hColor=15, so we tried with h=35, hColor=35
    restored_img = cv2.fastNlMeansDenoisingColored(distorted_img, None, h=35, hColor=35, templateWindowSize=7, searchWindowSize=21)
    
    # Convert the restored image to grayscale for OpenCV ORB
    gray = cv2.cvtColor(restored_img, cv2.COLOR_RGB2GRAY)
    
    # Detect keypoints and compute descriptors on the restored image
    kp, des = orb.detectAndCompute(gray, None)
    
    # Draw keypoints on top of the restored image
    img_with_kp = cv2.drawKeypoints(restored_img, kp, None, color=(0, 255, 0), flags=0)
    
    ax.imshow(img_with_kp)
    ax.axis("off")

plt.tight_layout()
plt.show()