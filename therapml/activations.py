import math
import torch
import torch.nn as nn


class MySigmoid(nn.Module):
    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-x))


class MyReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0)


class MyGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MySoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x_max = x.max(dim=self.dim, keepdim=True).values
        e = torch.exp(x - x_max)
        return e / e.sum(dim=self.dim, keepdim=True)


class MySwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, w1, w2, w3):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, x):
        gate = x @ self.w1.T
        up = x @ self.w3.T
        return (gate * torch.sigmoid(gate) * up) @ self.w2.T
