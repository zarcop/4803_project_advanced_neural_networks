from torch import nn


class AlexNet4803(nn.Module):


    def __init__(self, in_channels = 32, k_classes = 10):
        super().__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=5, padding = 2, stride = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =  3 , stride = 2, padding = 1),
            nn.LocalResponseNorm(size = 5),
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.ReLU(),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(),
            nn.Linear(),
            nn.Linear(),
        )

    
    def forward(self, x):
        x = self.convolutional_block(x)
        x = self.fully_connected_layer(x)
        return x

