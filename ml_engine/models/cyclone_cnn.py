import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CycloneIntensityModel(nn.Module):
    """
    ResNet18-based Convolutional Neural Network for Cyclone Intensity Regression.
    """
    def __init__(self, pretrained=True):
        super(CycloneIntensityModel, self).__init__()
        
        # Load pre-trained ResNet18 weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.base_model = resnet18(weights=weights)
        
        # Replace the final fully connected layer for regression (1 output: wind speed)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Single regression value output
        )

    def forward(self, x):
        return self.base_model(x)

print("Cyclone CNN model architecture defined successfully!")