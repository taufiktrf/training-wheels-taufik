import torch.nn as nn
from therapml.linear import MyLinear
from therapml.activations import MyReLU
from therapml.normalization import MyLayerNorm
from therapml.dropout import MyDropout


class CustomMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            MyLinear(784, 256),
            MyLayerNorm(256),
            MyReLU(),
            MyDropout(0.1),
            MyLinear(256, 128),
            MyLayerNorm(128),
            MyReLU(),
            MyDropout(0.1),
            MyLinear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
