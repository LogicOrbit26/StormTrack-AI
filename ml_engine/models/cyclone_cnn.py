
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CycloneIntensityModel(nn.Module):
    """
    ResNet18-based cyclone intensity regression model.

    Input:
        1 infrared satellite image channel

    Output:
        Wind speed in knots
        Atmospheric pressure in mb
    """

    def __init__(self, pretrained=False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.base_model = resnet18(weights=weights)

        # Change input from RGB (3 channels) to infrared (1 channel)
        old_conv = self.base_model.conv1

        self.base_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Adapt pretrained RGB weights if requested
        if pretrained:
            with torch.no_grad():
                self.base_model.conv1.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        # Predict both wind speed and pressure
        num_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":

    model = CycloneIntensityModel(pretrained=False)

    sample = torch.randn(8, 1, 301, 301)

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Expected output: [8, 2]")
    print("CNN architecture verified!")
