import torch
from torch.nn.functional import softmax
from therapml.rope import apply_rope


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / d_k**0.5

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    return softmax(scores, dim=-1) @ V


def causal_self_attention(wq, wk, wv, x, return_attn_weights=False):
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

    if return_attn_weights:
        return attn_weights
    return attn_weights @ V


def multihead_self_attention(d_model, num_heads, wq, wk, wv, wo, x):
    d_head = d_model // num_heads
    seq = x.shape[-2]

    Q = x @ wq.T
    K = x @ wk.T
    V = x @ wv.T

    # create the head dimension
    Q = Q.unflatten(-1, (num_heads, d_head)).transpose(-3, -2)
    K = K.unflatten(-1, (num_heads, d_head)).transpose(-3, -2)
    V = V.unflatten(-1, (num_heads, d_head)).transpose(-3, -2)

    # causal attention
    causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
    out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    # merge heads
    out = out.transpose(-3, -2).flatten(-2)

    # combine heads with output projection
    return out @ wo.T


def grouped_query_attention(
    d_model,
    num_heads,
    num_kv_heads,
    wq,
    wk,
    wv,
    wo,
    x,
    theta=None,
    token_positions=None,
):
    d_head = d_model // num_heads
    seq = x.shape[-2]

    Q = x @ wq.T
    K = x @ wk.T
    V = x @ wv.T

    # create head dimensions
    Q = Q.unflatten(-1, (num_heads, d_head)).transpose(-3, -2)
    K = K.unflatten(-1, (num_kv_heads, d_head)).transpose(-3, -2)
    V = V.unflatten(-1, (num_kv_heads, d_head)).transpose(-3, -2)

    # optional RoPE
    if theta is not None:
        if token_positions is None:
            token_positions = torch.arange(seq, device=x.device)
        Q = apply_rope(Q, token_positions, theta)
        K = apply_rope(K, token_positions, theta)

    # repeat KV heads to match query heads
    if num_heads != num_kv_heads:
        group = num_heads // num_kv_heads
        K = K.repeat_interleave(group, dim=-3)
        V = V.repeat_interleave(group, dim=-3)

    # causal attention
    causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
    out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    # merge heads and project output
    out = out.transpose(-3, -2).flatten(-2)
    return out @ wo.T
