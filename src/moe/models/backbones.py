import torch.nn as nn
from torch import flatten
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

class FeatureBackboneMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        #1 input channel, 8 output channels, kernel size 3, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        #non linearity
        self.relu1 = nn.ReLU()
        #first pooling layer with kernel size 2, stride 2 reduces image to (8,14,14)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #8 input channels, 16 output channels, kernel size 3, stride 1 padding 1
        self.conv2 = nn.Conv2d(in_channels= 8,out_channels= 16 , kernel_size= 3, stride= 1, padding= 1)
        #non linearity
        self.relu2 = nn.ReLU()
        #second pooling layer with kernel size 2, stride 2 reduces image to (16,7,7)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # 16 inputs, 32 outputs, kernel size 3, stride 1, padding 1
        self.conv3 = nn.Conv2d(in_channels= 16,out_channels= 32 , kernel_size= 3, stride= 1, padding= 1)
        #non linearity
        self.relu3 = nn.ReLU()
        #output netwrok we have 32 channels and an image that is (7,7)
        self.output = nn.Linear(32 * 7 * 7, 512)
        
        self.output_dim = 512

    def forward(self, x):
        #pass through the first convolution and relu and pooling layers
        x = self.pool1(self.relu1(self.conv1(x)))
        #pass through the second convolution and relu and pooling layers
        x = self.pool2(self.relu2(self.conv2(x)))
        #pass through the final convolution and relu
        x = self.relu3(self.conv3(x))
        #flatten all dimensions except batch dimension which is dimension 0 so we start at 1
        x = flatten(x, 1)
        #pass through our output layer
        x = self.output(x)
        return x