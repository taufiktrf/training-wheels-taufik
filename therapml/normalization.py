import torch.nn as nn
from torch import sqrt

eps = 1e-5


class MyLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        dims = tuple(range(1, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, correction=0)
        return (x - mean) / sqrt(var + eps) * gamma + beta


class MyRMSNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma):
        rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x / rms) * gamma


class MyResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
