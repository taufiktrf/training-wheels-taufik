import torch
from therapml.attention import scaled_dot_product_attention
from therapml.rope import apply_rope


class KVCachedAttention:
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        max_seq_len,
        theta,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_model // num_heads
        self.theta = theta

        self.wq = q_proj_weight
        self.wk = k_proj_weight
        self.wv = v_proj_weight
        self.wo = o_proj_weight

        # Cached tensors: (batch, num_kv_heads, cached_seq, d_head)
        self.k_cache = None
        self.v_cache = None

    def reset(self):
        """Clear cache state for a new generation sequence."""
        self.k_cache = None
        self.v_cache = None

    def __call__(self, x):
        batch, new_seq, _ = x.shape
        cache_len = 0 if self.k_cache is None else self.k_cache.shape[-2]
        positions = torch.arange(cache_len, cache_len + new_seq, device=x.device)

        # Step 1: Project new tokens and split heads
        Q = (
            (x @ self.wq.T)
            .unflatten(-1, (self.num_heads, self.d_head))
            .transpose(-3, -2)
        )
        K = (
            (x @ self.wk.T)
            .unflatten(-1, (self.num_kv_heads, self.d_head))
            .transpose(-3, -2)
        )
        V = (
            (x @ self.wv.T)
            .unflatten(-1, (self.num_kv_heads, self.d_head))
            .transpose(-3, -2)
        )

        # Step 2: Apply RoPE to new Q and K
        Q = apply_rope(Q, positions, self.theta)
        K = apply_rope(K, positions, self.theta)

        # Step 3: Append new K and V to cache
        if self.k_cache is None:
            self.k_cache, self.v_cache = K, V
        else:
            self.k_cache = torch.cat([self.k_cache, K], dim=-2)
            self.v_cache = torch.cat([self.v_cache, V], dim=-2)

        K_full, V_full = self.k_cache, self.v_cache

        # Step 4: Repeat KV heads if GQA
        if self.num_heads != self.num_kv_heads:
            group = self.num_heads // self.num_kv_heads
            K_full = K_full.repeat_interleave(group, dim=-3)
            V_full = V_full.repeat_interleave(group, dim=-3)

        # Step 5: Causal mask over full cache
        total_seq = K_full.shape[-2]
        if new_seq == 1:
            mask = None  # Single-token decode attends to all past tokens and itself
        else:
            # Query positions attend to all prior and current key positions
            mask = (
                positions[:, None] >= torch.arange(total_seq, device=x.device)[None, :]
            )

        # Step 6: Attention and output projection
        out = scaled_dot_product_attention(Q, K_full, V_full, mask=mask)
        out = out.transpose(-3, -2).flatten(-2)
        return out @ self.wo.T
