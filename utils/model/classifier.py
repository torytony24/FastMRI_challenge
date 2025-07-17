import torch.nn as nn
import torch.nn.functional as F

class AnatomyClassifier(nn.Module):
    def __init__(self, in_channels: int, num_anatomies: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_anatomies)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return F.softmax(logits, dim=1)