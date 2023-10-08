import torch
import torch.nn as nn

class SimpleCenterNet(nn.Module):
    def __init__(self):
        super(SimpleCenterNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.center_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1) # 1 channel for center heatmap
        )
        
        self.regression_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1) # 4 channels for bounding box (xmin, ymin, xmax, ymax)
        )
    
    def forward(self, x):
        features = self.features(x)
        center = self.center_head(features)
        regression = self.regression_head(features)
        
        return center, regression
