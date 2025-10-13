# src/moe/heads/base.py
import torch.nn as nn

class BaseHead(nn.Module):
    """Uniform interface so all heads behave the same."""
    head_name = "Base"

    def forward(self, h, return_gate: bool = False):
        raise NotImplementedError

    @staticmethod
    def pack(logits, probs=None, sel_idx=None, aux_loss=None, return_gate=False):
        """
        logits   : (B, C)
        probs    : (B, num_experts)  gating probabilities
        sel_idx  : (B, k)            selected expert indices
        aux_loss : extra loss (optional)
        """
        return (logits, probs, sel_idx, aux_loss) if return_gate else logits
