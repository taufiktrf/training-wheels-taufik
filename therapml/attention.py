from torch.nn.functional import softmax


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / d_k**0.5

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    return softmax(scores, dim=-1) @ V
