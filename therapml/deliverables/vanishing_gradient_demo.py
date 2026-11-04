from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from therapml.linear import MyLinear
from therapml.activations import MyReLU, MySigmoid

torch.manual_seed(42)

# dataset
X = torch.randn(512, 64)
y = torch.randn(512, 1)


# 10 layers using MyLinear
def build_mlp(activation_cls):
    layers = []
    for _ in range(9):
        layers.append(MyLinear(64, 64, bias=True))
        layers.append(activation_cls())
    layers.append(MyLinear(64, 1, bias=True))
    return nn.Sequential(*layers)


sigmoid_model = build_mlp(MySigmoid)
relu_model = build_mlp(MyReLU)

loss_fn = nn.MSELoss()

# First backward pass
loss_fn(sigmoid_model(X), y).backward()
loss_fn(relu_model(X), y).backward()

sigmoid_grads = [
    m.weight.grad.norm().item() for m in sigmoid_model if isinstance(m, MyLinear)
]
relu_grads = [
    m.weight.grad.norm().item() for m in relu_model if isinstance(m, MyLinear)
]

# Train both models for 150 steps
opt_sigmoid = torch.optim.SGD(sigmoid_model.parameters(), lr=0.01)
opt_relu = torch.optim.SGD(relu_model.parameters(), lr=0.01)

sigmoid_losses = []
relu_losses = []

for _ in range(150):
    opt_sigmoid.zero_grad()
    loss_s = loss_fn(sigmoid_model(X), y)
    loss_s.backward()
    opt_sigmoid.step()
    sigmoid_losses.append(loss_s.item())

    opt_relu.zero_grad()
    loss_r = loss_fn(relu_model(X), y)
    loss_r.backward()
    opt_relu.step()
    relu_losses.append(loss_r.item())

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

layers = list(range(1, 11))
ax1.plot(layers, sigmoid_grads, "ro-", label="Sigmoid", linewidth=2)
ax1.plot(layers, relu_grads, "bo-", label="ReLU", linewidth=2)
ax1.set_yscale("log")
ax1.set_xlabel("Layer (1 = Input, 10 = Output)")
ax1.set_ylabel("Weight Gradient Norm (Log Scale)")
ax1.set_title("Vanishing Gradient Across Layers")
ax1.set_xticks(layers)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

ax2.plot(sigmoid_losses, "r-", label="Sigmoid", linewidth=2)
ax2.plot(relu_losses, "b-", label="ReLU", linewidth=2)
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("MSE Loss")
ax2.set_title("Training Loss (10-Layer MLP)")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
save_path = Path(__file__).parent / "vanishing_gradient_demo.png"
plt.savefig(save_path, dpi=150)
plt.close()

print(f"Plot saved to: {save_path}")
