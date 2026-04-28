
from torch import nn

#altough this was designed for the a 32X32X1 dataset of MNIST
# I am doing my own tweak to make a similar architecture for the CIFAR-10 dataset.

class LeNet5(nn.Module):
    def init(self, in_channels = 32, k_classes = 10): # probably going to be 10 with CIFAR-10
        self.convolutional_layer_1 = nn.Conv2d(32,32,3), 
        self.pooling_layer = nn.AvgPool2d(),
        self.convolutional_layer_2 = nn.Conv2d()
        self.pooling_layer_2 = nn.AvgPool2d()
        self.fully_connected_layer = nn.Linear()
        self.flat_layer = nn.Flatten()
        self.fully_connected_layer_2 = nn.Linear()
        convolutional_block = nn.Sequential(
            self.convolutional_layer_1,
            self.pooling_layer,
            self.convolutional_layer_2,
            self.pooling_layer_2,
            self.fully_connected_layer,
            self.flat_layer,
            self.fully_connected_layer_2
        )
    





 
    