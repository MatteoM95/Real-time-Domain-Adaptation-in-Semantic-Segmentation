import torch
from torch import nn


# noinspection PyTypeChecker
class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(out_channels * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.leakyRelu(x)
        x = self.conv2(x)
        x = self.leakyRelu(x)
        x = self.conv3(x)
        x = self.leakyRelu(x)
        x = self.conv4(x)
        x = self.leakyRelu(x)
        x = self.conv5(x)

        return x
