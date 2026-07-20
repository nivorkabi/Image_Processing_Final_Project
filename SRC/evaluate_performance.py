import random
import numpy as np
import cv2
import torch
from datasets import load_dataset
import albumentations as A
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 1. Setup and Reproducibility
random.seed(7)
np.random.seed(7)

print("=== Starting Full Quantitative Evaluation (Fixed Labels) ===")

# 2. Load Dataset
ds = load_dataset("nateraw/ade20k-tiny", split="train")
idxs = random.sample(range(len(ds)), 4)
images = [ds[i]["image"] for i in idxs]
gt_masks = [ds[i]["label"] for i in idxs]

# 3. Initialize All Models & Distortions
print("Loading models...")
yolo_model = YOLO("yolov8n.pt")
seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Define Distortions using modern Albumentations parameters to prevent warnings
jpeg_distort = A.ImageCompression(quality_range=(5, 5), p=1.0)
noise_distort = A.GaussNoise(p=1.0)
lowlight_distort = A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.6), contrast_limit=(-0.4, -0.4), p=1.0)

# Helper function for Bounding Box IoU (for YOLO Recall)
def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

# Helper function for Low Light Enhancement
def enhance_low_light(img_np):
    gamma = 2.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(img_np, table)
    lab = cv2.cvtColor(brightened, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

# Initialize metric accumulators
yolo_metrics = {"distorted_recall": [], "enhanced_recall": []}
orb_metrics = {"distorted_ratio": [], "enhanced_ratio": []}
seg_metrics = {"clean_miou": [], "distorted_miou": [], "enhanced_miou": []}

# 4. Evaluation Loop over the 4 sample images
print("Evaluating samples...")
for img_pil, gt_mask in zip(images, gt_masks):
    img_clean = np.array(img_pil)
    h, w = img_clean.shape[:2]
    gt_mask_np = np.array(gt_mask)
    
    # ----------------------------------------------------
    # TASK 1: YOLOv8 Object Detection Evaluation
    # ----------------------------------------------------
    res_clean = yolo_model.predict(img_clean, conf=0.25, verbose=False)[0]
    clean_boxes = res_clean.boxes.xyxy.cpu().numpy() if res_clean.boxes is not None else []
    
    img_dist_jpeg = jpeg_distort(image=img_clean)["image"]
    img_enh_jpeg = cv2.bilateralFilter(img_dist_jpeg, d=9, sigmaColor=75, sigmaSpace=75)
    
    res_dist = yolo_model.predict(img_dist_jpeg, conf=0.25, verbose=False)[0]
    dist_boxes = res_dist.boxes.xyxy.cpu().numpy() if res_dist.boxes is not None else []
    
    res_enh = yolo_model.predict(img_enh_jpeg, conf=0.25, verbose=False)[0]
    enh_boxes = res_enh.boxes.xyxy.cpu().numpy() if res_enh.boxes is not None else []
    
    if len(clean_boxes) > 0:
        matched_dist = 0
        for cb in clean_boxes:
            if any(box_iou(cb, db) >= 0.5 for db in dist_boxes):
                matched_dist += 1
        yolo_metrics["distorted_recall"].append(matched_dist / len(clean_boxes))
        
        matched_enh = 0
        for cb in clean_boxes:
            if any(box_iou(cb, eb) >= 0.5 for eb in enh_boxes):
                matched_enh += 1
        yolo_metrics["enhanced_recall"].append(matched_enh / len(clean_boxes))
        
    # ----------------------------------------------------
    # TASK 2: ORB Feature Detection Evaluation
    # ----------------------------------------------------
    gray_clean = cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY)
    kp_clean, des_clean = orb.detectAndCompute(gray_clean, None)
    
    img_dist_noise = noise_distort(image=img_clean)["image"]
    gray_dist = cv2.cvtColor(img_dist_noise, cv2.COLOR_RGB2GRAY)
    kp_dist, des_dist = orb.detectAndCompute(gray_dist, None)
    
    img_enh_noise = cv2.fastNlMeansDenoisingColored(img_dist_noise, None, h=35, hColor=35, templateWindowSize=7, searchWindowSize=21)
    gray_enh = cv2.cvtColor(img_enh_noise, cv2.COLOR_RGB2GRAY)
    kp_enh, des_enh = orb.detectAndCompute(gray_enh, None)
    
    if des_clean is not None and len(kp_clean) > 0:
        if des_dist is not None:
            matches_dist = bf.match(des_clean, des_dist)
            orb_metrics["distorted_ratio"].append(len(matches_dist) / len(kp_clean))
        else:
            orb_metrics["distorted_ratio"].append(0.0)
            
        if des_enh is not None:
            matches_enh = bf.match(des_clean, des_enh)
            orb_metrics["enhanced_ratio"].append(len(matches_enh) / len(kp_clean))
        else:
            orb_metrics["enhanced_ratio"].append(0.0)

    # ----------------------------------------------------
    # TASK 3: SegFormer Semantic Segmentation Evaluation (mIoU with Fixed Shift)
    # ----------------------------------------------------
    img_dist_light = lowlight_distort(image=img_clean)["image"]
    img_enh_light = enhance_low_light(img_dist_light)
    
    def get_seg_pred(img):
        inputs = seg_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = seg_model(**inputs)
        upsampled = torch.nn.functional.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
        return upsampled.argmax(dim=1)[0].numpy()
        
    pred_clean = get_seg_pred(img_clean)
    pred_dist = get_seg_pred(img_dist_light)
    pred_enh = get_seg_pred(img_enh_light)
    
    # The fix: Align gt (1..150) with pred (0..149) by subtracting 1 from the comparison target
    def compute_miou(pred, gt):
        valid = gt > 0
        classes = np.unique(gt[valid])
        ious = []
        for c in classes:
            intersection = np.logical_and(pred == (c - 1), gt == c).sum()
            union = np.logical_or(pred == (c - 1), gt == c).sum()
            if union > 0:
                ious.append(intersection / union)
        return np.mean(ious) if len(ious) > 0 else 0.0

    seg_metrics["clean_miou"].append(compute_miou(pred_clean, gt_mask_np))
    seg_metrics["distorted_miou"].append(compute_miou(pred_dist, gt_mask_np))
    seg_metrics["enhanced_miou"].append(compute_miou(pred_enh, gt_mask_np))

# 5. Print Final Summary Table
print("\n" + "="*50)
print("FINAL QUANTITATIVE METRICS SUMMARY TABLE")
print("="*50)
print(f"| Vision Task | Metric | Baseline (Clean) | Distorted | Enhanced (Mitigation) |")
print(f"| :--- | :--- | :--- | :--- | :--- |")
print(f"| Object Detection (YOLOv8) | Recall (vs Baseline) | 1.000 | {np.mean(yolo_metrics['distorted_recall']):.3f} | {np.mean(yolo_metrics['enhanced_recall']):.3f} |")
print(f"| Feature Detection (ORB) | Matching Ratio | 1.000 | {np.mean(orb_metrics['distorted_ratio']):.3f} | {np.mean(orb_metrics['enhanced_ratio']):.3f} |")
print(f"| Segmentation (SegFormer) | mean IoU (vs GT) | {np.mean(seg_metrics['clean_miou']):.3f} | {np.mean(seg_metrics['distorted_miou']):.3f} | {np.mean(seg_metrics['enhanced_miou']):.3f} |")
print("="*50)