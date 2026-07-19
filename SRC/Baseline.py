import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset

# ==========================================
# 1. Configuration & Reproducibility
# ==========================================

# Set a fixed seed to ensure deterministic, reproducible image sampling
random.seed(7) 

# ==========================================
# 2. Dataset Loading (ADE20K Tiny)
# ==========================================

# Output loading status to the terminal
print("Loading dataset (ade20k-tiny)...")

# Download and load the training split of the lightweight ADE20K dataset
ds = load_dataset("nateraw/ade20k-tiny", split="train") 

# Get the total number of available images in the dataset
N = len(ds) 

# ==========================================
# 3. Data Sampling & Extraction
# ==========================================

# Randomly sample 4 unique, non-repeating image indices
idxs = random.sample(range(N), 4) 

# Extract complete data dictionary objects for the sampled indices
samples = [ds[i] for i in idxs] 

# Extract the clean, original source images (PIL format)
images = [s["image"] for s in samples] 

# Extract matching semantic segmentation target masks (Ground Truth)
masks = [s["label"] for s in samples] 

# ==========================================
# 4. Image Blending (Overlay) Function
# ==========================================

def overlay_mask(img_pil, mask_pil, alpha=0.45): #
    """
    Blends a semantic segmentation mask over the clean source image 
    using alpha blending configuration.
    """
    # Convert PIL source image to RGB and cast to float32 for numerical pixel operations
    img = np.array(img_pil.convert("RGB")).astype(np.float32) 
    
    # Convert target mask to an integer array representing category/class IDs
    m = np.array(mask_pil).astype(np.int32) 
    
    # Initialize a reproducible random state for class coloring
    rng = np.random.default_rng(0) 
    
    # Generate a fixed, randomized color palette for 256 possible classes
    palette = rng.integers(0, 255, size=(256, 3), dtype=np.uint8) 
    
    # Map integer class IDs to their corresponding RGB values from the generated palette
    color = palette[(m % 256).astype(np.int32)] 
    
    # Apply alpha blending formula, clip values to valid 0-255 range, and cast back to uint8
    out = (img * (1 - alpha) + color.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8) 
    
    # Convert the final calculated pixel matrix back into a PIL Image object
    return Image.fromarray(out) 

# ==========================================
# 5. Rendering Image Grid Visuals
# ==========================================

print("Plotting images with ground truth masks...")

# Initialize Matplotlib canvas with 4 subplots arranged in a single row
fig, axes = plt.subplots(1, 4, figsize=(20, 5)) 

# Iterate through subplots, source images, and target masks concurrently
for ax, img, m in zip(axes, images, masks): 
    # Render the alpha-blended color map image into the active subplot
    ax.imshow(overlay_mask(img, m)) 
    
    # Turn off coordinate grid ticks and bounding frames for a clean output format
    ax.axis("off") 

# Adjust margins automatically to optimize visual layout spacing
plt.tight_layout() 

# Launch interactive UI window on the desktop to display results
plt.show()