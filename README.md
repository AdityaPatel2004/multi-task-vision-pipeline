# Assignment 2: Multilabel Classification & Semantic Segmentation

## Submission Structure

You **MUST** follow this exact folder structure. Your submission will be evaluated by running your code live on the test set — **do not hardcode or pre-save predictions**.

## Submission Folder Structure

```
<Roll_no>_<name>_Assignment_2/
├── classification/
│   ├── model.py              # Must implement ClassificationModel class
│   └── weights/              # Your saved model weights
├── segmentation/
│   ├── model.py              # Must implement SegmentationModel class
│   └── weights/              # Your saved model weights
├── submission.csv             # Kaggle submission file
├── training_notebook.ipynb    # Your training code and analysis
├── statistical_tests.ipynb    # Statistical analysis (see below)
├── requirements.txt           # pip dependencies
└── report.pdf                 # Short report
```

---

## API Specification (MANDATORY)

### Task 1: Multilabel Classification

In `classification/model.py`, implement:

```python
import numpy as np
from PIL import Image

class ClassificationModel:
    def __init__(self, weights_dir: str):
        """
        Initialize model and load weights.
  
        Args:
            weights_dir: relative path to classification/weights/ folder
        """
        # Load your trained model here
        pass

    def predict(self, image: np.ndarray) -> dict:
        """
        Predict which of the 20 classes are present in the image.
  
        Args:
            image: RGB image as numpy array of shape (H, W, 3), dtype uint8
  
        Returns:
            dict mapping class_name (str) -> probability (float in [0, 1])
            Must contain ALL 20 classes:
                aeroplane, bicycle, bird, boat, bottle, bus, car, cat,
                chair, cow, diningtable, dog, horse, motorbike, person,
                pottedplant, sheep, sofa, train, tvmonitor
  
        Example:
            {"aeroplane": 0.02, "bicycle": 0.95, "person": 0.88, ...}
        """
        pass
```

### Task 2: Semantic Segmentation

In `segmentation/model.py`, implement:

```python
import numpy as np
from PIL import Image

class SegmentationModel:
    def __init__(self, weights_dir: str):
        """
        Initialize model and load weights.
    
        Args:
            weights_dir: absolute path to segmentation/weights/ folder
        """
        # Load your trained model here
        pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict semantic segmentation mask for the image.
    
        Args:
            image: RGB image as numpy array of shape (H, W, 3), dtype uint8
        
        Returns:
            Segmentation mask as numpy array of shape (H, W), dtype uint8
            Pixel values: 0 = background, 1-20 = class index, 255 = ignore
        
            Class index mapping:
                1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle,
                6=bus, 7=car, 8=cat, 9=chair, 10=cow,
                11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person,
                16=pottedplant, 17=sheep, 18=sofa, 19=train, 20=tvmonitor
        """
        pass
```

---

## Important Rules

1. **No hardcoded predictions.** Your model must perform actual inference.
2. **`predict()` must work on any image**, including ones not in the training set.
3. **Do not modify** the dataset or evaluation scripts.
4. **Model loading must be self-contained** — all weights go in the `weights/` folder.
5. Your code will be evaluated on the **test split** which you received without looking up ground truth.

## Dataset

The dataset is in `Dataset/` with this structure:

```
Dataset/
├── train/
│   ├── images/              # JPEG images for training
│   ├── annotations/         # XML annotations (bounding boxes + classes)  
│   ├── segmentation_masks/  # PNG semantic segmentation ground truth
│   └── labels.csv           # Multilabel binary matrix
├── test/
│   ├── images/              # JPEG images for testing
├── dataset_info.json
└── README.md
```

## Kaggle Leaderboard Submission

You must generate a `submission.csv` file containing predictions for both tasks on the test set. This file will be uploaded to Kaggle to evaluate your leaderboard score.

The CSV must have **exactly 3 columns**:

| Column | Description |
|--------|-------------|
| `image_id` | Image filename stem (e.g. `img_00005`) |
| `classification` | Space-separated list of predicted class names (threshold ≥ 0.5, or `background` if none) |
| `segmentation_rle` | Run-Length Encoding (RLE) of the predicted segmentation mask |

### Example `submission.csv`

```csv
image_id,classification,segmentation_rle
img_00005,person car,3 10 25 8 100 15
img_00009,cat,5 20 50 30
img_00010,person bicycle dog,2 5 12 18 40 22
```

### Formatting Details

- **Classification**: List only the class names your model predicts as present, separated by a single space. Use exact class names as listed in `dataset_info.json`. If no class is predicted, write `background`.
- **Segmentation**: Encode your 2D segmentation mask using Run-Length Encoding (RLE) on the row-major flattened mask. Format is space-separated triplets of `<start> <length> <class_value>` (e.g., `start1 length1 value1 start2 length2 value2...`). Only encode non-zero (foreground) pixels. Make sure the mask is resized to the original image dimensions before encoding.

## Evaluation Metrics

- **Classification**: mean F1-score
- **Segmentation**: mIoU (mean Intersection-over-Union), pixel accuracy, per-class IoU
