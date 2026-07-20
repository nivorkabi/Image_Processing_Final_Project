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
* **Distorted Inference (`src/yolo_distorted.py`):** Completed. Severe JPEG compression (Quality = 5) caused significant degradation, dropping bounding box confidence scores and lowering overall model recall due to grid artifacts.
* **Mitigation & Restoration (`src/yolo_mitigated.py`):** Completed. Applied classical **Bilateral Filtering** (d=9, sigmaColor=75, sigmaSpace=75) to smooth compression artifacts while preserving crucial structural edges.
* **Key Findings:** The Bilateral Filter successfully restored the model's performance without fine-tuning. Missed objects (like the sink) were re-detected (Confidence = 0.37), and degraded objects saw a major confidence recovery (e.g., TV frame bounding box confidence jumped from 0.41 to 0.71).

### 3. Feature Detection Tasks (ORB & Gaussian Noise)
* **Baseline Inference (`src/orb_baseline.py`):** Completed. Maintained stable keypoint detection on clear structural features like wall edges, corners, and object boundaries.
* **Distorted Inference (`src/orb_distorted.py`):** Completed. Injecting Gaussian Noise completely disrupted the algorithm. The local pixel-level intensity gradients introduced by the white noise caused ORB to abandon real structural corners and cluster heavily on random noise pixels.
* **Mitigation & Restoration (`src/orb_mitigated.py`):** Completed. Evaluated **Fast Non-Local Means (NLM) Denoising** under two hyperparameter configurations:
  * *Initial Run (h=15):* Insufficient noise suppression. Residual high-frequency micro-fluctuations remained, causing ORB to still mistake noise remnants for localized pixel gradients instead of physical corners.
  * *Optimized Run (h=35):* Achieved aggressive noise suppression. In well-exposed frames, the background noise was completely flattened into smooth surfaces, forcing ORB to successfully return to true structural features like the arched mirror boundaries.

### 4. Semantic Segmentation Tasks (SegFormer & Low Light)
* **Baseline Inference (`src/segformer_baseline.py`):** Completed. Generated clean and well-defined semantic segmentations mapping architectural features and indoor objects under baseline lighting conditions.
* **Distorted Inference (`src/segformer_distorted.py`):** Completed. Subjecting the images to severe low-light conditions (-60% brightness, -40% contrast) completely blinded the model, triggering a domain shift where the structural feature boundaries collapsed into a single uniform prediction block.
* **Mitigation & Restoration (`src/segformer_mitigated.py`):** Completed. Applied a classical pre-processing pipeline combining non-linear **Gamma Correction (gamma=2.2)** and local adaptive contrast enhancement via **CLAHE**.
* **Key Findings:** The classical enhancement failed to recover model performance, leading to a complete activation collapse (100% solid color output maps). While these filters successfully brightened features for human perception, the non-linear pixel manipulations severely distorted the underlying pixel distribution. This structural domain shift caused SegFormer's internal layer normalization layers to fail, forcing the network to classify the entire image frame as a single background category. This highlights that robust semantic segmentation in low-light scenarios requires deep domain adaptation or end-to-end network fine-tuning rather than traditional image processing filters.