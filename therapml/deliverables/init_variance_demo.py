from pathlib import Path
import matplotlib.pyplot as plt
import torch
from therapml.linear import kaiming_init, xavier_init, linear
from therapml.activations import MyReLU

torch.manual_seed(42)

# dimensions
n_layers = 50
dim = 256
batch_size = 64
relu = MyReLU()

# Kaiming Init + ReLU
x_kaiming = torch.randn(batch_size, dim)
kaiming_vars = [x_kaiming.var().item()]

for _ in range(n_layers):
    w = kaiming_init(fan_in=dim, fan_out=dim)
    x_kaiming = relu(linear(dim, dim, w, x_kaiming))
    kaiming_vars.append(x_kaiming.var().item())

# Xavier Init + ReLU
x_xavier_relu = torch.randn(batch_size, dim)
xavier_relu_vars = [x_xavier_relu.var().item()]

for _ in range(n_layers):
    w = xavier_init(fan_in=dim, fan_out=dim)
    x_xavier_relu = relu(linear(dim, dim, w, x_xavier_relu))
    xavier_relu_vars.append(x_xavier_relu.var().item())

# Xavier Init + Tanh
x_xavier_tanh = torch.randn(batch_size, dim)
xavier_tanh_vars = [x_xavier_tanh.var().item()]

for _ in range(n_layers):
    w = xavier_init(fan_in=dim, fan_out=dim)
    x_xavier_tanh = torch.tanh(linear(dim, dim, w, x_xavier_tanh))
    xavier_tanh_vars.append(x_xavier_tanh.var().item())


# Plot
plt.figure(figsize=(9, 5))
layers = list(range(n_layers + 1))

plt.plot(
    layers,
    kaiming_vars,
    "b-",
    linewidth=2.5,
    label="Kaiming Normal + ReLU (Target ~ 1.0)",
)
plt.plot(
    layers,
    xavier_tanh_vars,
    "g--",
    linewidth=2.0,
    label="Xavier Normal + Tanh (Stable)",
)
plt.plot(
    layers,
    xavier_relu_vars,
    "r-",
    linewidth=2.0,
    label="Xavier Normal + ReLU (Vanishing)",
)

plt.yscale("log")
plt.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Ideal Variance (1.0)")
plt.xlabel("Layer Depth (0 = Input, 50 = 50th Layer)")
plt.ylabel("Activation Variance (Log Scale)")
plt.title("Activation Variance Across 50 Layers: Kaiming vs. Xavier")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

save_path = Path(__file__).parent / "init_variance_demo.png"
plt.savefig(save_path, dpi=150)
plt.close()

print(f"\nPlot saved to: {save_path}")
