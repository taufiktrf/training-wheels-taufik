import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from therapml.bpe import BPETokenizer
from therapml.evaluation import perplexity
from therapml.loss import MyCrossEntropyLoss
from therapml.model import MiniLlama
from therapml.pt_optimizers import MyAdamW

# Configuration
steps = 500
batch_size = 32
context_length = 64
vocab_size = 512
cur_dir = Path(__file__).parent

# Dataset: TinyShakespeare tokenized with BPE
corpus = Path("tests/fixtures/tinyshakespeare.txt").read_text(encoding="utf-8")
tokenizer = BPETokenizer()
tokenizer.train(corpus[:50_000], vocab_size)

train_tokens = torch.tensor(tokenizer.encode(corpus[:300_000]), dtype=torch.long)
val_tokens = torch.tensor(
    tokenizer.encode(corpus[1_000_000:1_020_000]), dtype=torch.long
)


def get_batch(data):
    idx = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in idx])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in idx])
    return x, y


def train_variant(name, num_kv_heads):
    torch.manual_seed(42)
    model = MiniLlama(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        d_ff=344,
        tie_weights=True,
    )
    optimizer = MyAdamW(model.parameters(), lr=3e-3)
    criterion = MyCrossEntropyLoss()

    val_steps, val_ppls = [], []
    t0 = time.perf_counter()

    print(f"\nTraining {name} (num_kv_heads={num_kv_heads})...")
    model.train()
    for step in range(steps):
        if step % 100 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                val_x, val_y = get_batch(val_tokens)
                ppl = perplexity(model(val_x), val_y)
                val_steps.append(step)
                val_ppls.append(ppl)
            model.train()

        x, y = get_batch(train_tokens)
        loss = criterion(model(x).reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(
                f"  Step {step+1:3d}/{steps} | Loss: {loss.item():.3f} | Val PPL: {val_ppls[-1]:.2f}"
            )

    speed = steps / (time.perf_counter() - t0)
    return {"steps": val_steps, "ppls": val_ppls, "speed": speed}


# Run ablation: GQA (2 KV heads) vs. MHA (4 KV heads)
gqa = train_variant("GQA", num_kv_heads=2)
mha = train_variant("MHA", num_kv_heads=4)

print("\n--- Ablation Summary ---")
print(f"GQA: Final PPL = {gqa['ppls'][-1]:.2f}, Speed = {gqa['speed']:.2f} steps/s")
print(f"MHA: Final PPL = {mha['ppls'][-1]:.2f}, Speed = {mha['speed']:.2f} steps/s")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(gqa["steps"], gqa["ppls"], "o-", label="GQA (2 KV heads)", color="#1f77b4")
ax1.plot(mha["steps"], mha["ppls"], "s--", label="MHA (4 KV heads)", color="#ff7f0e")
ax1.set(title="Validation Perplexity vs. Steps", xlabel="Step", ylabel="Perplexity")
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend()

ax2.bar(
    ["GQA", "MHA"],
    [gqa["speed"], mha["speed"]],
    color=["#1f77b4", "#ff7f0e"],
    width=0.4,
)
ax2.set(title="Training Speed (steps/s)", ylabel="Steps / Second")
ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

fig.tight_layout()
fig.savefig(cur_dir / "ablation_results.png", dpi=150)
plt.close(fig)
print(f"Saved plot → {cur_dir / 'ablation_results.png'}")
