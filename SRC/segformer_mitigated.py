import random
import numpy as np
import torch
import cv2
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

# 3. Define Low-Light Distortion
transform = A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.6), contrast_limit=(-0.4, -0.4), p=1.0)

# 4. Function to apply Gamma Correction + CLAHE Enhancement
def enhance_low_light(img_np):
    """Applies Gamma correction to brighten and CLAHE to restore local contrast."""
    # Step A: Gamma Correction (gamma > 1 brightens the image)
    gamma = 2.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(img_np, table)
    
    # Step B: CLAHE on LAB Color Space to enhance edges without distorting colors
    lab = cv2.cvtColor(brightened, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

# 5. Load Pretrained SegFormer Model and Processor
print("Loading SegFormer model...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# 6. Process Pipeline and Run Inference
print("Processing images (Distortion -> Enhancement -> SegFormer)...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, img_pil in zip(axes, images):
    img_np = np.array(img_pil)
    
    # Apply low light distortion
    augmented = transform(image=img_np)
    distorted_img = augmented["image"]
    
    # Apply our classical enhancement method (Gamma + CLAHE)
    restored_img = enhance_low_light(distorted_img)
    
    # Preprocess the enhanced image for the model
    inputs = processor(images=restored_img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Rescale logits to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=img_pil.size[::-1], 
        mode="bilinear",
        align_corners=False
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Plot the restored segmentation map
    ax.imshow(pred_seg, cmap="tab20")
    ax.axis("off")

plt.tight_layout()
plt.show()