"""segmentation/model.py - ResNet101 + ASPP + UNet decoder (v5)"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv_r6 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv_r12 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv_r18 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.5))
    def forward(self, x):
        h, w = x.shape[2:]
        p = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([self.conv1x1(x), self.conv_r6(x), self.conv_r12(x), self.conv_r18(x), p], dim=1))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class SegResNet101UNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        backbone = models.resnet101(weights=None)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = ASPP(2048, 256)
        self.dec4 = DecoderBlock(256, 1024, 256)
        self.dec3 = DecoderBlock(256, 512, 128)
        self.dec2 = DecoderBlock(128, 256, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(256, num_classes, 1)
        )
    def forward(self, x):
        e0 = self.layer0(x)
        e1 = self.layer1(self.pool(e0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        x = self.aspp(e4)
        x = self.dec4(x, e3)
        x = self.dec3(x, e2)
        x = self.dec2(x, e1)
        x = self.dec1(x, e0)
        x = F.relu(self.final_up(x))
        return self.final_conv(x)

class SegmentationModel:
    def __init__(self, weights_dir: str):
        self.device = torch.device("cpu")
        self.model = SegResNet101UNet(num_classes=21)
        weights_path = os.path.join(weights_dir, "best_model.pth")
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        img = Image.fromarray(image)
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(img_t)
            pred = out.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST))
