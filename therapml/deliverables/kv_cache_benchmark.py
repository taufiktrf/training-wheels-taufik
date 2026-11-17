import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from therapml.attention import grouped_query_attention
from therapml.kv_cache import KVCachedAttention

# Benchmark Configuration
d_model = 512
num_heads = 8
num_kv_heads = 2
d_head = d_model // num_heads  # 64
batch_size = 4
theta = 10000.0
seq_lens = [128, 256, 512]
num_trials = 3

# Fixed weights for reproducibility
torch.manual_seed(42)
wq = torch.randn(d_model, d_model) / d_model**0.5
wk = torch.randn(num_kv_heads * d_head, d_model) / d_model**0.5
wv = torch.randn(num_kv_heads * d_head, d_model) / d_model**0.5
wo = torch.randn(d_model, d_model) / d_model**0.5


def benchmark_generation(seq_len, use_cache=True, trials=num_trials):
    x = torch.randn(batch_size, seq_len, d_model)

    def decode(n_steps):
        if use_cache:
            attn = KVCachedAttention(
                d_model, num_heads, num_kv_heads, seq_len, theta, wq, wk, wv, wo
            )
            for t in range(n_steps):
                attn(x[:, t : t + 1])
        else:
            for t in range(1, n_steps + 1):
                grouped_query_attention(
                    d_model,
                    num_heads,
                    num_kv_heads,
                    wq,
                    wk,
                    wv,
                    wo,
                    x[:, :t],
                    theta=theta,
                )

    # Warmup
    decode(min(8, seq_len))

    # Average elapsed time over trials
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        decode(seq_len)
        times.append(time.perf_counter() - t0)

    avg_elapsed = sum(times) / len(times)
    return (batch_size * seq_len) / avg_elapsed


print(
    f"\nKV-Cache Speed Benchmark (Batch={batch_size}, d_model={d_model}, {num_trials}-trial avg)"
)
print(
    f"{'Seq Length':<12} {'No Cache (tok/s)':<20} {'With Cache (tok/s)':<20} {'Speedup':<10}"
)
print("-" * 65)

no_cache_speeds = []
cached_speeds = []

with torch.no_grad():
    for s in seq_lens:
        slow = benchmark_generation(s, use_cache=False)
        fast = benchmark_generation(s, use_cache=True)
        no_cache_speeds.append(slow)
        cached_speeds.append(fast)
        speedup = fast / slow
        print(f"{s:<12} {slow:>14.1f}       {fast:>16.1f}       {speedup:>6.1f}x")

# Plot Generation Speed vs Sequence Length
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    seq_lens,
    cached_speeds,
    "o-",
    color="#1f77b4",
    linewidth=2,
    markersize=6,
    label="With KV-Cache (O(1) step)",
)
ax.plot(
    seq_lens,
    no_cache_speeds,
    "s--",
    color="#d62728",
    linewidth=2,
    markersize=6,
    label="Without Cache (O(N) step recompute)",
)

for s, fast, slow in zip(seq_lens, cached_speeds, no_cache_speeds):
    ax.annotate(
        f"{fast / slow:.1f}x",
        (s, fast),
        textcoords="offset points",
        xytext=(0, 9),
        ha="center",
        fontweight="bold",
        color="#1f77b4",
    )

ax.set_title(
    f"Autoregressive Generation Speed: With vs. Without KV-Cache\n(Batch={batch_size}, d_model={d_model}, GQA-4, {num_trials}-trial avg)"
)
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Generation Speed (tokens / second)")
ax.set_xticks(seq_lens)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()

fig.tight_layout()
fig.savefig(Path(__file__).parent / "kv_cache_benchmark.png", dpi=150)
plt.close(fig)
print("\nSaved plot → therapml/deliverables/kv_cache_benchmark.png")
