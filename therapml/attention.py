import torch
from torch.nn.functional import softmax


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / d_k**0.5

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    return softmax(scores, dim=-1) @ V


def causal_self_attention(wq, wk, wv, x):
    Q = x @ wq.T
    K = x @ wk.T
    V = x @ wv.T

    # raw attention scores
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / d_k**0.5

    # upper triangular mask
    seq = x.shape[-2]
    upper_tri_mask = torch.triu(
        torch.full((seq, seq), float("-inf"), device=x.device), diagonal=1
    )
    scores = scores + upper_tri_mask

    # Softmax and weighted sum of values
    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ V
