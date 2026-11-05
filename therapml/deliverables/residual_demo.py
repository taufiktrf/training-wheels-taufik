from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from therapml.linear import MyLinear
from therapml.activations import MyReLU
from therapml.normalization import MyResidualBlock
from therapml.pt_optimizers import MySGD
from therapml.loss import MyCrossEntropyLoss

torch.manual_seed(42)

# Synthetic data (20-layer)
dim = 32
X = torch.randn(500, dim)
y = (X[:, 0] * X[:, 1] + X[:, 2] ** 2 > 0.5).long()
loss_fn = MyCrossEntropyLoss()


# Build plain and residual models
def make_model(residual=False):
    layers = [MyLinear(dim, dim, bias=True), MyReLU()]
    for _ in range(18):
        block = nn.Sequential(MyLinear(dim, dim, bias=True), MyReLU())
        layers.append(MyResidualBlock(block) if residual else block)
    layers.append(MyLinear(dim, 2, bias=True))
    return nn.Sequential(*layers)


plain_model = make_model(residual=False)
res_model = make_model(residual=True)

# Scale to prevent variance compounding at initialization
for m in res_model.modules():
    if isinstance(m, MyResidualBlock):
        m.fn[0].weight.data.mul_(0.1)


# Train for 200 epochs
def train(model):
    opt = MySGD(model.parameters(), lr=0.1)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    acc = (model(X).argmax(dim=-1) == y).float().mean().item() * 100
    return losses, acc


plain_losses, plain_acc = train(plain_model)
res_losses, res_acc = train(res_model)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(
    plain_losses,
    "r-",
    label=f"20-Layer Plain (Acc: {plain_acc:.1f}%)",
    linewidth=2,
)
plt.plot(
    res_losses,
    "b-",
    label=f"20-Layer Residual (Acc: {res_acc:.1f}%)",
    linewidth=2,
)
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("20-Layer Network: Plain vs. Residual Loss Curves")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

save_path = Path(__file__).parent / "residual_demo.png"
plt.savefig(save_path, dpi=150)
plt.close()
