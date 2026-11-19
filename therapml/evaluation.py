import math
import torch
from therapml.loss import MyCrossEntropyLoss
from therapml.activations import MySoftmax


def perplexity(logits, targets):
    vocab_size = logits.shape[-1]
    loss = MyCrossEntropyLoss()(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return math.exp(loss.item())


def generate(model, prompt_ids, max_new_tokens, temperature, context_length):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        try:
            device = next(model.buffers()).device
        except StopIteration:
            device = torch.device("cpu")

    running = list(prompt_ids)
    new_tokens = []
    softmax = MySoftmax(dim=-1)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop to context length
            ctx_ids = running[-context_length:]
            idx = torch.tensor([ctx_ids], dtype=torch.long, device=device)

            logits = model(idx)
            next_logits = logits[0, -1, :]

            # greedy (deterministic) vs temperature sampling
            if temperature == 0.0:
                next_token = int(next_logits.argmax().item())
            else:
                probs = softmax(next_logits / temperature)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            running.append(next_token)
            new_tokens.append(next_token)

    return new_tokens
