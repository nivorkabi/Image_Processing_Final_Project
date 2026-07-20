import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 1. Configuration & Reproducibility
random.seed(7)

# 2. Load the exact same 4 clean images
print("Loading dataset...")
ds = load_dataset("nateraw/ade20k-tiny", split="train")
idxs = random.sample(range(len(ds)), 4)
images = [ds[i]["image"] for i in idxs]

# 3. Load Pretrained SegFormer Model and Processor
print("Loading SegFormer model...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# 4. Run Inference on Clean Images
print("Running SegFormer inference on clean images...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, img_pil in zip(axes, images):
    # Preprocess the image
    inputs = processor(images=img_pil, return_tensors="pt")
    
    # Run prediction without gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Rescale logits to match the original image size (Height, Width)
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=img_pil.size[::-1], 
        mode="bilinear",
        align_corners=False
    )
    
    # Get the predicted class label for each pixel
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Plot the segmentation map using a colormap
    ax.imshow(pred_seg, cmap="tab20")
    ax.axis("off")

plt.tight_layout()
plt.show()