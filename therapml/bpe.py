import re
from collections import Counter

# GPT-2-style: contractions, optional-space + word, punctuation, whitespace
SPLIT_RE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""", re.IGNORECASE
)


def _apply_merge(seq, a, b, new_id):
    """Replace every adjacent (a, b) in seq with new_id."""
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.special_tokens = {}

    def train(
        self, corpus: str, vocab_size: int, special_tokens: list[str] | None = None
    ):
        chunks = (
            re.split("|".join(re.escape(s) for s in special_tokens), corpus)
            if special_tokens
            else [corpus]
        )
        word_freqs = Counter(
            tuple(w.encode("utf-8")) for c in chunks for w in SPLIT_RE.findall(c)
        )

        while len(self.vocab) < vocab_size - len(special_tokens or []):
            # Count every adjacent pair, weighted by word frequency
            pair_freqs = Counter(
                pair
                for w, freq in word_freqs.items()
                for pair in zip(w, w[1:])
                for _ in range(freq)
            )
            if not pair_freqs:
                break

            # Merge the most frequent pair into a new token
            a, b = max(pair_freqs, key=lambda p: pair_freqs[p])
            new_id = len(self.vocab)
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            word_freqs = {
                _apply_merge(w, a, b, new_id): freq for w, freq in word_freqs.items()
            }

        # Append special tokens to the end of the vocabulary
        for s in special_tokens or []:
            sid = len(self.vocab)
            self.special_tokens[s] = sid
            self.vocab[sid] = s.encode("utf-8")

        return self

    def encode(self, text: str) -> list[int]:
        parts = (
            re.split(f"({'|'.join(re.escape(s) for s in self.special_tokens)})", text)
            if self.special_tokens
            else [text]
        )
        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue

            for word in SPLIT_RE.findall(part):
                tokens = tuple(word.encode("utf-8"))
                while len(tokens) >= 2:
                    best = min(
                        (p for p in zip(tokens, tokens[1:]) if p in self.merges),
                        key=lambda p: self.merges[p],
                        default=None,
                    )
                    if not best:
                        break
                    tokens = _apply_merge(tokens, *best, self.merges[best])
                ids.extend(tokens)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return b"".join(self.vocab.get(i, b"") for i in token_ids).decode(
            "utf-8", errors="replace"
        )
