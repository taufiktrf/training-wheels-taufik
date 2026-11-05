import torch.nn as nn
from torch import zeros_like, rand_like


class MyDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x, training=True):
        if not training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return zeros_like(x)
        mask = (rand_like(x.float()) >= self.p).to(x.dtype)
        return x * mask / (1.0 - self.p)
