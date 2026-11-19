from pathlib import Path
import matplotlib.pyplot as plt
import torch
import wandb

from therapml.bpe import BPETokenizer
from therapml.evaluation import generate, perplexity
from therapml.loss import MyCrossEntropyLoss
from therapml.model import MiniLlama
from therapml.pt_optimizers import MyAdamW
from therapml.training import clip_gradients, lr_schedule

# Configuration
steps = 2000
batch_size = 32
context_length = 64
vocab_size = 512
cur_dir = Path(__file__).parent

# tokenize TinyShakespeare
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


# Model & optimizer
torch.manual_seed(42)
model = MiniLlama(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=128,
    num_layers=4,
    num_heads=4,
    num_kv_heads=2,
    d_ff=344,
    tie_weights=True,
)
optimizer = MyAdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
criterion = MyCrossEntropyLoss()

# W&B offline tracking
wandb.init(
    project="therapml-training-taufik", name="evaluation-and-monitoring", mode="offline"
)

layer_norms = {"embed": [], **{f"block_{i}": [] for i in range(4)}}
samples = []

print(f"Training and Evaluating Mini-Llama ({steps} steps)...")

for step in range(steps):
    # Metric 1 & 2: Evaluate perplexity and generate samples at step 0 & every 500 steps
    if step == 0 or (step + 1) % 500 == 0:
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch(val_tokens)
            ppl = perplexity(model(val_x), val_y)
            print(f"\nStep {step} | Validation Perplexity: {ppl:.2f}")

            prompt = tokenizer.encode("KING HENRY:\n")
            for t in [0.1, 0.7, 1.0]:
                out = generate(
                    model,
                    prompt,
                    max_new_tokens=100,
                    temperature=t,
                    context_length=context_length,
                )
                text = tokenizer.decode(prompt + out)
                samples.append(f"Step {step} (T={t}) | PPL: {ppl:.2f}\n{text}\n")
                print(f"  [T={t}] {text[:60]}...")

            wandb.log({"step": step, "val_perplexity": ppl})
        model.train()

    # Training step
    lr = lr_schedule(step, steps, max_lr=3e-3, min_lr=3e-4)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_tokens)
    logits = model(x)
    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()

    # Record gradient norms per layer before clipping
    layer_norms["embed"].append(model.token_embeddings.weight.grad.norm().item())
    for i, block in enumerate(model.layers):
        b_norm = (
            sum(p.grad.norm() ** 2 for p in block.parameters() if p.grad is not None)
            ** 0.5
        )
        layer_norms[f"block_{i}"].append(b_norm.item())

    clip_gradients(model.parameters(), max_norm=1.0)
    optimizer.step()

    wandb.log({"step": step, "loss": loss.item(), "lr": lr})

    if (step + 1) % 100 == 0:
        print(f"Step {step+1}/{steps} | Loss: {loss.item():.3f} | LR: {lr:.2e}")

# Save qualitative generation samples
samples_file = cur_dir / "evaluation_generation_samples.txt"
samples_file.write_text("\n".join(samples), encoding="utf-8")

# Gradient Health per layer plot
fig, ax = plt.subplots(figsize=(8, 5))
for name, norms in layer_norms.items():
    ax.plot(norms, label=name, alpha=0.7)
ax.axhline(1.0, color="red", linestyle="--", label="Clip threshold (1.0)")
ax.set(
    title="Metric 3: Gradient Health (Per-Layer Norm)",
    xlabel="Step",
    ylabel="Grad Norm",
)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()

plot_file = cur_dir / "gradient_health.png"
fig.savefig(plot_file, bbox_inches="tight")
plt.close(fig)

wandb.log({"gradient_health": wandb.Image(str(plot_file))})
wandb.finish()

print(f"\nDone! Saved samples → {samples_file}")
print(f"Saved plot → {plot_file}")
