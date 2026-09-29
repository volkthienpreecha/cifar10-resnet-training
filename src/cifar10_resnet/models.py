"""Neural-network architectures used in the CIFAR-10 experiments."""

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn


class FCNN(nn.Module):
    """A configurable fully connected image-classification baseline."""

    def __init__(self, layer_dims: Sequence[int]) -> None:
        super().__init__()
        if len(layer_dims) < 2 or any(dimension <= 0 for dimension in layer_dims):
            raise ValueError("layer_dims must contain at least two positive dimensions")
        self.layers = nn.ModuleList(
            nn.Linear(input_dim, output_dim)
            for input_dim, output_dim in pairwise(layer_dims)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(inputs, 1)
        for layer in self.layers[:-1]:
            features = torch.relu(layer(features))
        return self.layers[-1](features)


class SimpleCNN(nn.Module):
    """A compact LeNet-style convolutional baseline for 32×32 RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class ResidualBlock(nn.Module):
    """Two-layer residual block with an optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channel counts must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = torch.relu(self.bn1(self.conv1(inputs)))
        residual = self.bn2(self.conv2(residual))
        return torch.relu(residual + self.shortcut(inputs))


class ResNet18(nn.Module):
    """A CIFAR-10-sized ResNet-18 built from :class:`ResidualBlock`."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.in_channels = 64
        self.layer1 = self._make_stage(64, stride=1)
        self.layer2 = self._make_stage(128, stride=2)
        self.layer3 = self._make_stage(256, stride=2)
        self.layer4 = self._make_stage(512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def _make_stage(self, out_channels: int, stride: int) -> nn.Sequential:
        first = ResidualBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return nn.Sequential(first, ResidualBlock(out_channels, out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.layer4(self.layer3(self.layer2(self.layer1(features))))
        return self.classifier(torch.flatten(self.pool(features), 1))


class TunedCNN(nn.Module):
    """The tuned convolutional network used for the strongest recorded run."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.features = nn.Sequential(
            self._stage(3, 64),
            self._stage(64, 128),
            self._stage(128, 256),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _stage(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``model``."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
