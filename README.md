# Multi-Task Visual Understanding Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg?logo=pytorch)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

An end-to-end Computer Vision pipeline designed to extract dense semantic information from images. This project implements state-of-the-art deep learning architectures to simultaneously solve two core visual understanding tasks: **Multi-label Classification** and **Semantic Segmentation**. 

Unlike standard ML projects, this pipeline emphasizes **statistical rigor** by employing non-parametric tests to formally evaluate model performance and guarantee that improvements are statistically significant.

---

## 🚀 Key Features

* **Semantic Segmentation**: Pixel-level prediction across 20 diverse real-world object classes (e.g., vehicles, animals, indoor items) handling complex occlusions and object boundaries.
* **Multi-Label Classification**: Robust classification models capable of identifying multiple non-mutually exclusive, co-occurring objects within a single image context.
* **Rigorous Statistical Validation**: Moving beyond point estimates (like simple Accuracy or F1 scores) by implementing:
  * **Wilcoxon Signed-Rank Test** for comparing model performances pairwise.
  * **Bootstrap Confidence Intervals** to estimate the true population performance and assess model variance.
* **Dimensionality Reduction (PCA)**: Used for data analysis and feature extraction to handle high-dimensional image inputs effectively.
* **Modular Architecture**: Clean, modular Object-Oriented design separated into modular domains (`classification/` and `segmentation/`) for extensible deployment.

---

## 🧠 Technical Architecture

The project is logically divided into distinct modules, adhering to strict API specifications designed for automated deployment and testing:

1. **Classification Pipeline (`classification/model.py`)** 
   - Takes dynamic input sizes, processes standard RGB `uint8` images, and outputs calibrated probability confidence mappings for 20 unique classes.
2. **Segmentation Pipeline (`segmentation/model.py`)** 
   - Dedicated dense-prediction architecture returning 2D segmentation masks. Implements specific boundary-aware background/foreground separation mapping all 20 classes + ignore regions.
3. **Statistical Framework (`statistical_tests.ipynb`)**
   - Pure statistical assessment engine avoiding data dredging to definitively answer if model $A$ outperforms model $B$ using rigorous non-parametric hypothesis testing.

---

## 📊 Evaluation Metrics

Models in this project are optimized and evaluated across multiple stringent metrics to ensure qualitative and quantitative excellence:

* **Classification**: Monitored via **Mean F1-Score** (to penalize class imbalance and reward true positives across multi-class overlapping inputs).
* **Segmentation**: Evaluated using **mIoU** (mean Intersection over Union), **Pixel Accuracy**, and **Per-class IoU**.

---

## 🛠️ Usage & Setup

### Prerequisites
* Python 3.9+
* PyTorch & standard scientific packages

```bash
# Clone the repository
git clone https://github.com/AdityaPatel2004/multi-task-vision-pipeline.git
cd multi-task-vision-pipeline

# Install dependencies
pip install -r requirements.txt
```

### 📦 Dataset & Pre-trained Weights

Due to GitHub's file size limitations, the raw multi-task dataset and the pre-trained `.pth` models (approx 600MB total) are hosted externally.

**[Download the Dataset and Weights from Google Drive]((Insert your Google Drive Link Here))**

After downloading, extract the contents into the project root so your final structure looks like this:

```
multi-task-vision-pipeline/
├── Dataset/                            # Put the extracted dataset here
├── classification/
│   ├── model.py
│   └── weights/
│       └── best_model.pth              # Put the 95MB classification weights here
├── segmentation/
│   ├── model.py
│   └── weights/
│       └── best_model.pth              # Put the 256MB segmentation weights here
...
```

---

## 🤝 Conclusion

This project serves as a comprehensive demonstration of applied Computer Vision, proving the ability to not just build complex deep learning models, but effectively orchestrate, evaluate, and statistically defend their validity.
