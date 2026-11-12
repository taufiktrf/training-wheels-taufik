from pathlib import Path
import matplotlib.pyplot as plt

# Model configuration (LLaMA-style)
d_model = 4096
num_heads = 32
d_head = d_model // num_heads  # 128
batch_size = 16
bytes_per_elem = 2  # fp16 / bf16

seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
configs = {
    "MHA (32 KV heads)": 32,
    "GQA-8 (8 KV heads)": 8,
    "GQA-4 (4 KV heads)": 4,
    "MQA (1 KV head)": 1,
}


# Calculate KV-cache memory in Megabytes: 2 (K & V) * batch * seq * kv_heads * d_head * bytes
def kv_cache_memory_mb(num_kv_heads, seq_len):
    total_bytes = 2 * batch_size * seq_len * num_kv_heads * d_head * bytes_per_elem
    return total_bytes / (1024**2)


memory_data = {name: [] for name in configs}
for seq in seq_lens:
    row = f"{seq:<12} "
    for name, kv_heads in configs.items():
        mb = kv_cache_memory_mb(kv_heads, seq)
        memory_data[name].append(mb)
        row += f"{mb:>10.1f} MB         "

# Plot memory comparison
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]

for (name, mem_list), color in zip(memory_data.items(), colors):
    ax.plot(
        seq_lens, mem_list, "o-", label=name, color=color, linewidth=2, markersize=5
    )

ax.set_title(
    f"KV-Cache Memory: MHA vs. GQA vs. MQA\n(Batch={batch_size}, d_model={d_model}, 16-bit precision)"
)
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("KV-Cache Memory (MB)")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()

fig.tight_layout()
fig.savefig(Path(__file__).parent / "gqa_memory_comparison.png", dpi=150)
plt.close(fig)
