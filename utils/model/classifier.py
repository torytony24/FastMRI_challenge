import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomyClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 16, 320, 184]
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 32, 160, 92]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 64, 1, 1]
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

