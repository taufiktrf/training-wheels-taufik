from pathlib import Path
from collections import Counter
import re
import matplotlib.pyplot as plt

from therapml.bpe import BPETokenizer

VOCAB_SIZES = [256, 1000, 5000, 10_000, 32_000]

corpus = Path("tests/fixtures/tinyshakespeare.txt").read_text(encoding="utf-8")
train = corpus[:50_000]
heldout = corpus[1_000_000:1_020_000]
sentences = [l for l in heldout.splitlines() if l.strip()]
rare = [
    "Supercalifragilisticexpialidocious",
    "antidisestablishmentarianism",
    "Pneumonoultramicroscopic",
]
common = [w for w, _ in Counter(re.findall(r"[a-z]+", train)).most_common(20)]

ratios, avg_toks = [], []
print(f"\n{'V':<8} {'bytes/tok':<12} {'tok/sent':<12} {'rare tok':<12} {'common tok'}")
print("-" * 58)

for v in VOCAB_SIZES:
    tok = BPETokenizer().train(train, v)
    encode = tok.encode

    ratio = len(heldout.encode()) / len(encode(heldout))
    avg_sent = sum(len(encode(s)) for s in sentences) / len(sentences)
    rare_avg = sum(len(encode(" " + w)) for w in rare) / len(rare)
    comm_avg = sum(len(encode(" " + w)) for w in common) / len(common)

    ratios.append(ratio)
    avg_toks.append(avg_sent)
    print(f"{v:<8} {ratio:<12.3f} {avg_sent:<12.2f} {rare_avg:<12.2f} {comm_avg:.2f}")

# dual-axis plot: compression ratio + avg tokens/sentence
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

(l1,) = ax1.plot(
    VOCAB_SIZES, ratios, "o-", color="#1f77b4", lw=2, ms=6, label="bytes / token"
)
(l2,) = ax2.plot(
    VOCAB_SIZES,
    avg_toks,
    "s--",
    color="#ff7f0e",
    lw=2,
    ms=6,
    label="avg tokens / sentence",
)

ax1.set(
    xlabel="Vocabulary Size",
    ylabel="Compression Ratio (bytes / token)",
    title="BPE: Compression Ratio vs. Vocabulary Size\n(trained on 50 KB TinyShakespeare)",
    xscale="log",
)
ax2.set_ylabel("Avg Tokens per Sentence", color="#ff7f0e")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
ax1.legend(handles=[l1, l2], loc="center right")
ax1.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(Path(__file__).parent / "vocab_size_experiment.png", dpi=150)
plt.close(fig)
print("\nPlot saved → therapml/deliverables/vocab_size_experiment.png")
