import torch
import torch.nn as nn

from therapml.activations import MySigmoid
from therapml.attention import grouped_query_attention
from therapml.linear import MyLinear
from therapml.normalization import MyRMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.theta = theta

        # pre-attention RMSNorm
        self.ln1 = MyRMSNorm(d_model)

        # GQA projections
        d_head = d_model // num_heads
        self.attn = nn.Module()
        self.attn.q_proj = MyLinear(d_model, d_model)
        self.attn.k_proj = MyLinear(d_model, num_kv_heads * d_head)
        self.attn.v_proj = MyLinear(d_model, num_kv_heads * d_head)
        self.attn.output_proj = MyLinear(d_model, d_model)

        # pre-FFN RMSNorm
        self.ln2 = MyRMSNorm(d_model)

        # SwiGLU FFN projections
        self.ffn = nn.Module()
        self.ffn.w1 = MyLinear(d_model, d_ff)
        self.ffn.w2 = MyLinear(d_ff, d_model)
        self.ffn.w3 = MyLinear(d_model, d_ff)

    def forward(self, x, token_positions=None):
        # 1. Attention with pre-norm and residual
        normed_x = self.ln1(x)
        attn_out = grouped_query_attention(
            self.d_model,
            self.num_heads,
            self.num_kv_heads,
            self.attn.q_proj.weight,
            self.attn.k_proj.weight,
            self.attn.v_proj.weight,
            self.attn.output_proj.weight,
            normed_x,
            theta=self.theta,
            token_positions=token_positions,
        )
        h = x + attn_out

        # 2. SwiGLU FFN with pre-norm and residual
        normed_h = self.ln2(h)
        gate = self.ffn.w1(normed_h)
        up = self.ffn.w3(normed_h)
        ffn_out = self.ffn.w2(gate * MySigmoid()(gate) * up)

        return h + ffn_out


class MiniLlama(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        num_kv_heads,
        d_ff,
        rope_theta=10000.0,
        tie_weights=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.tie_weights = tie_weights

        # token embedding table
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, num_kv_heads, d_ff, theta=rope_theta
                )
                for _ in range(num_layers)
            ]
        )

        # final RMSNorm and LM head
        self.ln_final = MyRMSNorm(d_model)
        self.lm_head = MyLinear(d_model, vocab_size)

        # tie embedding and output projection weights
        if tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

        # standard llama weight initialization (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, MyLinear, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, MyRMSNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, in_indices, token_positions=None):
        batch, seq = in_indices.shape
        x = self.token_embeddings(in_indices)

        if token_positions is None:
            token_positions = torch.arange(seq, device=in_indices.device)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)


def parameter_audit(model):
    total_params = sum(p.numel() for p in model.parameters())
    rows = []
    print(f"\n{'Parameter':<45} {'Shape':<25} {'Count':>12} {'Share (%)':>10}")
    print("-" * 95)
    for name, p in model.named_parameters():
        count = p.numel()
        percent = (count / total_params) * 100.0
        rows.append(
            {
                "name": name,
                "shape": tuple(p.shape),
                "num_params": count,
                "percent": percent,
            }
        )
        print(f"{name:<45} {str(tuple(p.shape)):<25} {count:>12,d} {percent:>9.2f}%")
    print("-" * 95)
    print(f"{'Total Parameters':<45} {'':<25} {total_params:>12,d} {100.0:>9.2f}%\n")
    return rows
