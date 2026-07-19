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

### 2. Object Detection Baseline Inference (`src/yolo_baseline.py`)
* **Status:** Completed
* **Description:** Integrated the pretrained `yolov8n.pt` (Nano) model from Ultralytics. Conducted baseline inference on the clean sample images to establish the initial bounding boxes and object classification scores, which serve as our baseline pseudo-ground truth.

---

## Next Computational Steps
1. **Apply Digital Distortion:** Introduce severe JPEG compression artifacts to the image batch.
2. **Distorted Inference:** Quantify YOLOv8 performance degradation on the corrupted images.
3. **Restoration & Mitigation:** Apply classical Bilateral Filtering to mitigate compression artifacts.