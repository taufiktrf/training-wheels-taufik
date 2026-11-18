import math
import torch


def lr_schedule(step, total_steps, max_lr, min_lr, warmup_frac=0.1):
    warmup_steps = int(warmup_frac * total_steps)

    # warmup
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # cosine decay
    if step <= total_steps:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)

    return min_lr


def clip_gradients(parameters, max_norm):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    # global L2 norm across all parameters
    total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in params))

    # scale gradients in-place
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.mul_(clip_coef)

    return float(total_norm)
