# Image_Processing_Final_Project
Evaluating the robustness of computer vision models,  under image distortions, and mitigating degradation using pre-processing enhancement and fine-tuning.
# Evaluating the Robustness of Image Processing and Vision Icons

## Project Overview
This project evaluates the robustness of various image processing and computer vision algorithms under common environmental and digital distortions. We analyze performance degradation across low-level and high-level vision tasks and investigate two mitigation approaches: image enhancement (pre-processing) and model fine-tuning.

---

## Dataset
* **Selected Dataset:** `ADE20K-Tiny` (loaded directly via HuggingFace `datasets`).
* **Ground Truth:** Includes semantic segmentation maps. For object detection and feature matching, baseline clean image inferences serve as the pseudo-ground truth.

---

## Project Choices & Configuration

The project implements 3 distinct vision tasks, subjected to 3 different distortions, and evaluated using specific robustness metrics:

| Vision Task | Algorithm / Model | Evaluation Metric | Image Distortion | Enhancement Method |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Detection** | **ORB** (OpenCV) | Good Matches / Clean Keypoints Ratio | **GaussNoise** (Albumentations) | Fast Non-Local Means (NLM) |
| **Object Detection** | **YOLOv8** (Ultralytics) | Mean Detection Recall | **Severe JPEG Compression** | Bilateral Filtering |
| **Semantic Segmentation** | **SegFormer** (NVIDIA) | Mean Intersection over Union (mIoU) | **Low Light** (Brightness Contrast) | Gamma Correction + CLAHE |

---

## Experimental Progress & Milestones

### 1. Dataset Exploration & Baseline Visualization (`src/baseline.py`)
* **Status:** Completed
* **Description:** Successfully configured the data pipeline to load `ade20k-tiny`. Implemented an alpha-blending overlay function to visualize the 4 sample clean images alongside their official semantic segmentation ground-truth masks.

### 2. Object Detection Tasks (YOLOv8 & JPEG Compression)
* **Baseline Inference (`src/yolo_baseline.py`):** Completed. Established baseline pseudo-ground truth on clean images.
* **Distorted Inference (`src/yolo_distorted.py`):** Completed. Severe JPEG compression (Quality = 5) caused significant degradation, dropping bounding box confidence scores (e.g., from 0.41 to blind/missed detections) and lowering overall model recall due to grid artifacts.
* **Mitigation & Restoration (`src/yolo_mitigated.py`):** Completed. Applied classical **Bilateral Filtering** ($d=9, \sigma_{Color}=75, \sigma_{Space}=75$) to smooth compression artifacts while preserving crucial structural edges.
* **Key Findings:** The Bilateral Filter successfully restored the model's performance without fine-tuning. In distorted frames, missed objects (like the sink) were re-detected (Confidence = 0.37), and degraded objects saw a major confidence recovery (e.g., TV frame bounding box confidence jumped from 0.41 to 0.71).

---

## Next Computational Steps
1. **Feature Detection Pipeline:** Set up the ORB feature tracking baseline.
2. **Noise Application:** Inject Gaussian Noise using Albumentations.
3. **Low-Level Restoration:** Evaluate Fast Non-Local Means (NLM) filtering to recover keypoint match ratios.