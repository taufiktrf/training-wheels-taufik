import argparse
import os
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

# Run W&B in offline mode (saves logs locally without account or login)
os.environ["WANDB_MODE"] = "offline"

# 1. Parse CLI Arguments
parser = argparse.ArgumentParser(description="Train Mini-Llama on TinyShakespeare")
parser.add_argument("--steps", type=int, default=600, help="Total training steps")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--max_lr", type=float, default=3e-3, help="Max learning rate")
parser.add_argument("--min_lr", type=float, default=3e-4, help="Min learning rate")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# 2. Load Dataset & Train BPE Tokenizer
data_path = Path("tests/fixtures/tinyshakespeare.txt")
text = (
    data_path.read_text(encoding="utf-8")
    if data_path.exists()
    else "Default training text " * 5000
)
train_text = text[:300_000]
val_text = text[1_000_000:1_020_000] if len(text) > 1_020_000 else text[300_000:320_000]

vocab_size = 512
tokenizer = BPETokenizer()
tokenizer.train(
    train_text[:50_000],
    vocab_size,
    special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
)

train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
val_tokens = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

# 3. Model & Optimizer Setup
context_length = 64
model = MiniLlama(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=128,
    num_layers=4,
    num_heads=4,
    num_kv_heads=2,
    d_ff=344,
    tie_weights=True,
).to(device)

optimizer = MyAdamW(
    model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1
)
criterion = MyCrossEntropyLoss()

# 4. Initialize W&B (Offline mode, no account or login needed)
wandb.init(
    project="therapml-training-taufik",
    name="mini-llama-training",
    mode="offline",
    config={
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": 128,
        "num_layers": 4,
    },
)


def get_batch(data, batch_size):
    starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[s : s + context_length] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + context_length + 1] for s in starts]).to(device)
    return x, y


print(
    f"\nStarting Mini-Llama Training (Device={device.upper()}, Steps={args.steps}, Batch={args.batch_size})"
)
print(f"{'Step':<8} {'Loss':<12} {'LR':<12} {'Grad Norm':<12}")
print("-" * 45)

# History tracking for plotting and logs
history = {
    "losses": [],
    "lrs": [],
    "grad_norms": [],
    "val_steps": [],
    "val_ppls": [],
    "samples": [],
}

# 5. Training Loop
model.train()
for step in range(args.steps):
    # Adjust learning rate (10% warmup + 90% cosine decay)
    lr = lr_schedule(step, args.steps, args.max_lr, args.min_lr)
    for group in optimizer.param_groups:
        group["lr"] = lr

    # Forward pass
    x, y = get_batch(train_tokens, args.batch_size)
    logits = model(x)
    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

    # Backward pass & gradient clipping
    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_gradients(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Track metrics
    history["losses"].append(loss.item())
    history["lrs"].append(lr)
    history["grad_norms"].append(grad_norm)

    # Log to W&B
    wandb.log(
        {
            "step": step,
            "train_loss": loss.item(),
            "lr": lr,
            "grad_norm": grad_norm,
        }
    )

    if step % 100 == 0 or step == args.steps - 1:
        print(f"{step:<8d} {loss.item():<12.3f} {lr:<12.2e} {grad_norm:<12.3f}")

    # Qualitative generation & validation perplexity evaluation
    if (step > 0 and step % 200 == 0) or step == args.steps - 1:
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch(val_tokens, args.batch_size)
            val_logits = model(val_x)
            val_ppl = perplexity(val_logits, val_y)

            history["val_steps"].append(step)
            history["val_ppls"].append(val_ppl)

            # Generate sample at T=0.7
            prompt = tokenizer.encode("KING HENRY:\n")
            sample_ids = generate(
                model,
                prompt,
                max_new_tokens=60,
                temperature=0.7,
                context_length=context_length,
            )
            sample_text = tokenizer.decode(prompt + sample_ids)

            sample_entry = f"Step {step} | Val PPL: {val_ppl:.2f}\n{sample_text}\n" + (
                "-" * 40
            )
            history["samples"].append(sample_entry)

            print(f"  --> Step {step:4d} | Val Perplexity: {val_ppl:.2f}")
            print(f"  --> Sample:\n{sample_text[:120]}...\n")

            wandb.log({"val_perplexity": val_ppl, "step": step})

        model.train()

# 6. Save Generation Samples to logs/
samples_file = logs_dir / "generation_samples.txt"
samples_file.write_text("\n\n".join(history["samples"]), encoding="utf-8")

# 7. Plot Training Curves (Loss, LR, Grad Norm, Val Perplexity)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Loss curve
axes[0, 0].plot(history["losses"], color="#1f77b4", label="Train Loss")
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Cross-Entropy Loss")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Learning rate schedule
axes[0, 1].plot(history["lrs"], color="#ff7f0e", label="Learning Rate")
axes[0, 1].set_title("Learning Rate Schedule (Warmup + Cosine)")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("LR")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Gradient norm
axes[1, 0].plot(history["grad_norms"], color="#2ca02c", label="Pre-clip Grad Norm")
axes[1, 0].axhline(1.0, color="red", linestyle="--", label="Clip threshold (1.0)")
axes[1, 0].set_title("Gradient Health (L2 Norm)")
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Norm")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Validation Perplexity
if history["val_ppls"]:
    axes[1, 1].plot(
        history["val_steps"],
        history["val_ppls"],
        "o-",
        color="#d62728",
        label="Val Perplexity",
    )
    axes[1, 1].set_title("Validation Perplexity")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Perplexity")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

fig.tight_layout()
plot_path = logs_dir / "training_curves.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

# 8. Log Final Artifacts to W&B
wandb.log({"training_curves": wandb.Image(str(plot_path))})
wandb.finish()

print(f"\nTraining Complete!")
print(f"- Saved training curves plot → {plot_path}")
print(f"- Saved text generation samples → {samples_file}")
print(f"- Logged all steps to local Weights & Biases (offline mode)")
