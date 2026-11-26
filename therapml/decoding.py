import torch
from therapml.activations import MySigmoid, MySoftmax
from therapml.kv_cache import KVCachedAttention


def _make_caches(model):
    return [
        KVCachedAttention(
            b.d_model,
            b.num_heads,
            b.num_kv_heads,
            model.context_length,
            b.theta,
            b.attn.q_proj.weight,
            b.attn.k_proj.weight,
            b.attn.v_proj.weight,
            b.attn.output_proj.weight,
        )
        for b in model.layers
    ]


def _forward_cached(model, in_indices, caches):
    x = model.token_embeddings(in_indices)
    for b, cache in zip(model.layers, caches):
        x = x + cache(b.ln1(x))
        h = b.ln2(x)
        gate = b.ffn.w1(h)
        x = x + b.ffn.w2(gate * MySigmoid()(gate) * b.ffn.w3(h))
    return model.lm_head(model.ln_final(x))


def top_k_filter(logits, k):
    if k >= logits.shape[-1]:
        return logits.clone()
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    if p >= 1.0:
        return logits.clone()
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    probs = MySoftmax(dim=-1)(sorted_logits)
    # keep tokens whose strictly higher-ranked mass is < p (rank 0 is always kept)
    keep = (torch.cumsum(probs, dim=-1) - probs) < p
    keep[..., 0] = True
    keep = torch.zeros_like(keep).scatter(-1, order, keep)
    return logits.masked_fill(~keep, float("-inf"))


def sample_next_token(logits, temperature, top_k=None, top_p=None):
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = MySoftmax(dim=-1)(logits)
    # multinomial accepts only 1D/2D tensors, so flatten leading dims temporarily
    tokens = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).squeeze(-1)
    return tokens.reshape(logits.shape[:-1])


def generate(
    model,
    prompt_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    context_length=64,
    use_kv_cache=False,
):
    t = next(iter(model.state_dict().values()), None)
    device = t.device if t is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    tokens = list(prompt_ids)
    caches = _make_caches(model) if use_kv_cache else None

    with torch.no_grad():
        for step in range(max_new_tokens):
            # prefill prompt on step 0
            ctx = [tokens[-1]] if use_kv_cache and step else tokens[-context_length:]
            inp = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = _forward_cached(model, inp, caches) if use_kv_cache else model(inp)
            next_tok = sample_next_token(
                logits[0, -1], temperature, top_k, top_p
            ).item()
            tokens.append(next_tok)

    return tokens[len(prompt_ids) :]
