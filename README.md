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
| **Semantic Segmentation** | **SegFormer** (NVIDIA) | mean Intersection over Union (mIoU) | **Low Light** (Brightness Contrast) | Gamma Correction + CLAHE |

---

## Quantitative Performance Summary

The table below presents the final quantitative evaluation across all three vision pipelines under clean, distorted, enhanced (mitigated), and fine-tuned states:

| Vision Task | Metric | Baseline (Clean) | Distorted | Enhanced (Mitigation) | Fine-Tuned Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Object Detection (YOLOv8)** | Recall (vs Baseline) | 1.000 | 0.067 | 0.633 | 0.067 |
| **Feature Detection (ORB)** | Matching Ratio | 1.000 | 0.442 | 0.364 | N/A (Classical Task) |
| **Segmentation (SegFormer)** | mean IoU (vs GT) | 0.532 | 0.000 | 0.000 | N/A (Pending Step) |

---

## Experimental Progress & Visual Milestones

### 1. Dataset Exploration & Baseline Visualization (`src/baseline.py`)
Successfully configured the data pipeline to load `ade20k-tiny` and implemented alpha-blending to visualize the baseline clean images alongside their official segmentation masks.

![Dataset Baseline](dataset_baseline.png)

### 2. Object Detection Tasks (YOLOv8 & JPEG Compression)
Severe JPEG compression completely degraded structural features, dropping model Recall to a critical **0.067**. Pre-processing with a **Bilateral Filter** successfully recovered broken edges, restoring the Recall significantly to **0.633**.

* **Baseline Clean Detections:**
![YOLO Baseline](yolo_baseline.png)
* **Distorted (JPEG Compressed Quality=5):**
![YOLO Distorted](yolo_distorted.png)
* **Mitigated (Bilateral Filter Restored):**
![YOLO Mitigated](yolo_mitigated.png)

### 3. Feature Detection Tasks (ORB & Gaussian Noise)
Gaussian Noise disrupted localized pixel gradients, dropping ORB matching accuracy to **0.442**. Aggressive NLM filtering ($h=35$) cleared the background noise visually and returned keypoints to physical objects, but the resulting pixel smoothing altered the descriptor distributions, yielding a matching ratio of **0.364**.

* **Baseline Clean Keypoints:**
![ORB Baseline](orb_baseline.png)
* **Distorted (Gaussian Noise Applied):**
![ORB Distorted](orb_distorted.png)
* **Initial Denoising Attempt (NLM with h=15):**
![ORB Mitigated h15](orb_mitigated_h15.png)
* **Optimized Denoising (NLM with h=35):**
![ORB Mitigated h35](orb_mitigated_h35.png)

### 4. Semantic Segmentation Tasks (SegFormer & Low Light)
The baseline model achieved a **0.532 mIoU** against the Ground Truth. Severe low-light distortion caused a total activation collapse (**0.000 mIoU**). Classical illumination filtering (Gamma + CLAHE) failed to recover performance (**0.000 mIoU**) due to non-linear distribution warping that disrupted the network's internal layer normalizations.

* **Baseline Clean Segmentation Maps:**
![SegFormer Baseline](segformer_baseline.png)
* **Distorted (Severe Low Light Applied):**
![SegFormer Distorted](segformer_distorted.png)
* **Mitigated (Gamma Correction + CLAHE):**
![SegFormer Mitigated](segformer_mitigated.png)

### 5. Model Fine-Tuning & Evaluation Pipeline (Weeks 10-11)
* **Status:** Completed
* **Description:** Evaluated the performance of the fine-tuned YOLOv8 model weights (`runs/detect/train/weights/best.pt`) against traditional pre-processing filters under severe JPEG distortions.
* **Key Findings:** The fine-tuned model yielded a Recall of **0.067**, failing to improve upon the unmitigated distorted baseline. Under severe dataset size constraints (4 images) and a short training envelope (3 epochs), the model executed too few gradient updates to adjust its feature representations. This highlights that deep domain adaptation requires extensive target-domain data and training time, whereas classical Bilateral Filtering achieves an immediate block-artifact smoothing effect based on static mathematical priors.