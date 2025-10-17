import torch.nn as nn
from torchvision.models import resnet18

class FeatureBackbone(nn.Module):
    """ResNet-18 adapted for CIFAR-10, outputs 512-dim features."""
    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Identity()
        self.backbone = m
        self.output_dim = 512
    def forward(self, x):
        return self.backbone(x)
