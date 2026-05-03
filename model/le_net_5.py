
from torch import nn

#altough this was designed for the a 32X32X1 dataset of MNIST
# I am doing my own tweak to make a similar architecture for the CIFAR-10 dataset.

# this works both with CIFAR-10 and MNIST datasets.

class LeNet5(nn.Module):
    def __init__(self, in_channels = 32, k_classes = 10): # probably going to be 10 with CIFAR-10
        #convolutional block
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten()
        )
        #fully connected blocks
        self.fully_connected_block  =  nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, k_classes)
        )
    def forward_pass(self,x):
        x = self.convolutional_block(x)
        x = self.fully_connected_block(x)
        return x
       
    





 
    