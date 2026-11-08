import torch.nn as nn
from torch import sqrt, ones, zeros

eps = 1e-5


class MyLayerNorm(nn.Module):
    def __init__(self, shape=None):
        super().__init__()
        self.gamma = nn.Parameter(ones(shape)) if shape else None
        self.beta = nn.Parameter(zeros(shape)) if shape else None

    def forward(self, x, gamma=None, beta=None):
        g = self.gamma if gamma is None else gamma
        b = self.beta if beta is None else beta
        dims = tuple(range(1, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, correction=0)
        return (x - mean) / sqrt(var + eps) * g + b


class MyRMSNorm(nn.Module):
    def __init__(self, shape=None):
        super().__init__()
        self.gamma = nn.Parameter(ones(shape)) if shape else None

    def forward(self, x, gamma=None):
        g = self.gamma if gamma is None else gamma
        rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x / rms) * g


class MyResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
