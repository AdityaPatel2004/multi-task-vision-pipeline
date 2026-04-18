"""classification/model.py - ResNet50 + GeM + custom head (v5)"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

CLS_SIZE = 256

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)

class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.head(self.gem(self.features(x)))

CLASS_NAMES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
               "chair","cow","diningtable","dog","horse","motorbike","person",
               "pottedplant","sheep","sofa","train","tvmonitor"]

class ClassificationModel:
    def __init__(self, weights_dir: str):
        self.device = torch.device("cpu")
        self.model = MultiLabelClassifier(num_classes=20)
        weights_path = os.path.join(weights_dir, "best_model.pth")
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((CLS_SIZE + 32, CLS_SIZE + 32)),
            transforms.CenterCrop(CLS_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        thresh_path = os.path.join(weights_dir, "thresholds.npy")
        if os.path.exists(thresh_path):
            self.thresholds = np.load(thresh_path)
        else:
            self.thresholds = np.full(20, 0.5)

    def predict(self, image: np.ndarray) -> dict:
        img = Image.fromarray(image)
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(img_t)
            probs = torch.sigmoid(out).squeeze().cpu().numpy()
        return {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
