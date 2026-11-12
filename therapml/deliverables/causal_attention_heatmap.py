import torch
import matplotlib.pyplot as plt
from pathlib import Path
from therapml.attention import causal_self_attention

torch.manual_seed(42)

seq, d_in, d_k, d_v = 12, 32, 16, 24

x = torch.randn(1, seq, d_in)
wq = torch.randn(d_k, d_in) / d_in**0.5
wk = torch.randn(d_k, d_in) / d_in**0.5
wv = torch.randn(d_v, d_in) / d_in**0.5

# Call causal_self_attention
attn_weights = causal_self_attention(wq, wk, wv, x, return_attn_weights=True)
attn_matrix = attn_weights[0].detach()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(attn_matrix, cmap="Blues", vmin=0, vmax=1)
fig.colorbar(im, ax=ax, label="Attention Weight")
ax.set_title("Causal Self-Attention Weights\n(lower triangle only — no future leakage)")
ax.set_xlabel("Key Position")
ax.set_ylabel("Query Position")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "causal_attention_heatmap.png", dpi=150)
plt.close(fig)
print("Saved → therapml/deliverables/causal_attention_heatmap.png")
