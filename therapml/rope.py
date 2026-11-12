import torch


def apply_rope(x, token_positions, theta=10000.0):
    d_k = x.shape[-1]
    # Frequencies: theta_i = theta ** (-2i / d_k)
    idx = torch.arange(0, d_k, 2, device=x.device, dtype=torch.float32)
    freqs = theta ** (-idx / d_k)

    # Broadcast token_positions against the sequence dimension of x
    pos = token_positions.to(dtype=torch.float32)
    while pos.ndim < x.ndim - 1:
        pos = pos.unsqueeze(-2)
    angles = pos.unsqueeze(-1) * freqs

    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)

    # rotation
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out
