import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from therapml.loss import MyCrossEntropyLoss
from therapml.model import MiniLlama
from therapml.pt_optimizers import MyAdamW

# Benchmark Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
seq_len = 128
vocab_size = 512
d_model = 256
num_layers = 4
num_heads = 4
num_kv_heads = 2
d_ff = 688
n_steps = 30
n_warmup = 5

torch.manual_seed(42)

# Synthetic batch for training speed benchmark
x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
criterion = MyCrossEntropyLoss()


def benchmark_precision(dtype, use_scaler=False):
    torch.manual_seed(42)
    model = MiniLlama(
        vocab_size=vocab_size,
        context_length=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        d_ff=d_ff,
    ).to(device)
    optimizer = MyAdamW(model.parameters(), lr=1e-3)
    scaler = (
        torch.amp.GradScaler("cuda")
        if (use_scaler and device == "cuda")
        else None
    )

    def step():
        optimizer.zero_grad()
        with torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=(dtype != torch.float32),
        ):
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # Warmup
    for _ in range(n_warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed run
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return n_steps / elapsed


# Evaluate supported precisions
configs = [
    ("FP32", torch.float32, False),
    ("BF16", torch.bfloat16, False),
]
if device == "cuda":
    configs.append(("FP16", torch.float16, True))

print(
    f"\nMixed-Precision Training Benchmark (Device={device.upper()}, Batch={batch_size}, seq_len={seq_len})"
)
print(f"{'Precision':<15} {'Speed (steps/s)':<20} {'Speedup':<10}")
print("-" * 50)

names = []
speeds = []
base_speed = None

for name, dtype, use_scaler in configs:
    spd = benchmark_precision(dtype, use_scaler=use_scaler)
    if base_speed is None:
        base_speed = spd
    speedup = spd / base_speed
    names.append(name)
    speeds.append(spd)
    print(f"{name:<15} {spd:>12.2f} steps/s {speedup:>8.2f}x")

# Plot Comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(names, speeds, color=["#1f77b4", "#2ca02c", "#ff7f0e"][: len(names)], width=0.5)

for bar, spd in zip(bars, speeds):
    height = bar.get_height()
    rel = spd / base_speed
    ax.annotate(
        f"{spd:.1f} steps/s\n({rel:.2f}x)",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

ax.set_title(
    f"Training Speed: FP32 vs. Mixed Precision\n(Device={device.upper()}, Batch={batch_size}, d_model={d_model})"
)
ax.set_ylabel("Training Speed (steps / second)")
ax.set_ylim(0, max(speeds) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(Path(__file__).parent / "mixed_precision_benchmark.png", dpi=150)
plt.close(fig)
print("\nSaved plot → therapml/deliverables/mixed_precision_benchmark.png")
