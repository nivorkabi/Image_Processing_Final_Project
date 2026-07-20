import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
import albumentations as A
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 1. Configuration & Reproducibility
random.seed(7)

# 2. Load the exact same 4 clean images
print("Loading dataset...")
ds = load_dataset("nateraw/ade20k-tiny", split="train")
idxs = random.sample(range(len(ds)), 4)
images = [ds[i]["image"] for i in idxs]

# 3. Define Low-Light Distortion using Albumentations
# We drastically reduce brightness and contrast to simulate a dark environment
transform = A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.6), contrast_limit=(-0.4, -0.4), p=1.0)

# 4. Load Pretrained SegFormer Model and Processor
print("Loading SegFormer model...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# 5. Apply distortion and run inference
print("Applying low-light distortion and running SegFormer...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, img_pil in zip(axes, images):
    img_np = np.array(img_pil)
    
    # Apply low light distortion
    augmented = transform(image=img_np)
    distorted_img = augmented["image"]
    
    # Preprocess the distorted image
    inputs = processor(images=distorted_img, return_tensors="pt")
    
    # Run prediction without gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Rescale logits to match the original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=img_pil.size[::-1], 
        mode="bilinear",
        align_corners=False
    )
    
    # Get the predicted class label for each pixel
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Plot the distorted segmentation map
    ax.imshow(pred_seg, cmap="tab20")
    ax.axis("off")

plt.tight_layout()
plt.show()