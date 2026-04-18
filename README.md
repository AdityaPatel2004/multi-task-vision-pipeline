# Multi-Task Classification and Semantic Segmentation Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg?logo=pytorch)

## Overview

This repository contains the implementation of an end-to-end computer vision pipeline designed for simultaneous multi-label image classification and semantic segmentation. The system predicts and segments 20 object classes, prioritizing robust model evaluation through formal statistical significance testing rather than relying solely on point estimates.

## Methodology

1. **Multi-Label Classification (`classification/`)**
   - Formulated to predict the presence of 20 non-mutually exclusive object classes within standard RGB imagery.
   - Performance optimized and evaluated using Mean F1-Score strategies to account for dataset imbalances.

2. **Semantic Segmentation (`segmentation/`)**
   - Dense prediction architecture outputting 2D segmentation masks for background and foreground class boundaries.
   - Evaluated across mean Intersection over Union (mIoU), pixel accuracy, and per-class IoU metrics.

3. **Statistical Significance Testing (`statistical_tests.ipynb`)**
   - Applies the **Wilcoxon Signed-Rank Test** for rigorous pairwise evaluation of model variations.
   - Utilizes **Bootstrap Confidence Intervals** to estimate population variance and true model performance bounds.

4. **Analysis**
   - Incorporates Principal Component Analysis (PCA) for dimensionality reduction and feature extraction visualization.

## Repository Structure

```text
.
├── classification/          # Classification model definition and inference API
├── segmentation/            # Segmentation model definition and inference API
├── training_notebook.ipynb  # Primary training loop, data pipelines, and loss formulations
├── statistical_tests.ipynb  # Non-parametric testing scripts
└── requirements.txt         # Environment dependencies
```

## Setup and Inference

### Requirements

Ensure a Python 3.9+ runtime environment is active.

```bash
git clone https://github.com/AdityaPatel2004/multi-task-vision-pipeline.git
cd multi-task-vision-pipeline
pip install -r requirements.txt
```

### Dataset and Pre-trained Models

Due to GitHub's file storage limits, the annotated datasets and pre-trained state dictionaries (`.pth`) are hosted independently.

**[Download Dataset and Model Weights](https://drive.google.com/drive/folders/1MFpQVgAqFX2hu9FSoIQQ08TZMoSMlE-9?usp=sharing)**

To prevent filename conflicts in the external Drive, model weights are prefixed by their respective domains. Extract the dataset into the root directory and rename the weight files strictly as outlined below prior to execution:

```text
multi-task-vision-pipeline/
├── Dataset/                            # Extracted dataset files
├── classification/
│   ├── model.py
│   └── weights/
│       └── best_model.pth              # Rename from classification_best_model.pth
├── segmentation/
│   ├── model.py
│   └── weights/
│       └── best_model.pth              # Rename from segmentation_best_model.pth
...
```
