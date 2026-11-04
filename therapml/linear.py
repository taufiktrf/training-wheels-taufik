import math
import torch
import torch.nn as nn


def linear(d_in, d_out, weights, x):
    return x @ weights.T


def kaiming_init(fan_in, fan_out):
    return torch.randn(fan_out, fan_in) * math.sqrt(2.0 / fan_in)


def xavier_init(fan_in, fan_out):
    return torch.randn(fan_out, fan_in) * math.sqrt(2.0 / (fan_in + fan_out))


class MyLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(kaiming_init(d_in, d_out))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.bias = None

    def forward(self, x):
        out = linear(self.d_in, self.d_out, self.weight, x)
        if self.bias is not None:
            out = out + self.bias
        return out
