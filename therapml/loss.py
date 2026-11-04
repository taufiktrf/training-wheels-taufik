import torch
import torch.nn as nn


class MyCrossEntropyLoss(nn.Module):
    def forward(self, logits, ground_truth):
        # log space arithmetic
        x_max = logits.max(dim=-1, keepdim=True).values
        shifted = logits - x_max
        log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))

        # indexed class
        if ground_truth.dim() == 1 or (
            ground_truth.dim() == 2 and ground_truth.shape[-1] == 1
        ):
            targets = (
                ground_truth.squeeze(-1).long()
                if ground_truth.dim() == 2
                else ground_truth.long()
            )
            nll = -log_probs[torch.arange(logits.shape[0]), targets]
            return nll.mean()
        else:
            # one hot case
            return -(ground_truth * log_probs).sum(dim=-1).mean()


class MyKLDivergenceLoss(nn.Module):
    def forward(self, input_log_probs, target_probs):
        kl = target_probs * (torch.log(target_probs) - input_log_probs)
        return kl.sum() / input_log_probs.shape[0]
