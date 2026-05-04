import torch
from torch import nn


class AlexNet4803(nn.Module):

    def __init__(self, in_channels=1, k_classes=10, k1=11, s1=4, p1=2):

        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=k1, stride=s1, padding=p1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fully_connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, k_classes),
        )

    def forward(self, x):
        x = self.convolutional_block(x)
        x = self.avgpool(x)
        x = self.fully_connected_layer(x)
        return x
