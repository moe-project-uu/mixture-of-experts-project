# src/moe/heads/soft_moe.py
from .base import BaseHead
import torch
import torch.nn as nn
import torch.nn.functional as F

class _make_expert(nn.Module):
    """
    Simple expert that maps backbone features directly to class logits:
      in_dim -> hidden -> ReLU -> num_classes
    Keeping experts shallow here is intentional for stability & speed.
    """
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, in_dim) -> logits: (B, num_classes)
        return self.net(h)

class SoftMoEHead(BaseHead):
    """
    Soft MoE head for single-vector inputs (post GlobalAvgPool).

    Forward:
      h: (B, D)  pooled features, D = 512
      returns:
        - if return_gate=False: logits (B, C)
        - if return_gate=True : (logits, probs, sel_idx=None, aux_loss=None)

    Notes:
      - Every input activates all experts (soft weights).
      - Compute cost ~ num_experts * expert-MLP.
      - 'probs' gives you per-sample expert usage for analysis. 
      i.e. it's the probability of each expert being used for each sample.
    """
    head_name = "SoftMoE"

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_experts: int = 4,
        hidden_mult: float = 2.0,
        temperature: float = 1.0,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert temperature > 0.0, "temperature must be > 0"
        self.num_experts = int(num_experts)
        self.hidden_mult = float(hidden_mult)
        self.temperature = float(temperature)
        self.dropout_p = float(dropout_p)

        # Simple linear gate: (B, D) -> (B, E)
        self.gate = nn.Linear(in_dim, self.num_experts, bias=True) # shape (512, num_experts)

        # Bank of expert MLPs: each maps (B, D) -> (B, C)
        self.experts = nn.ModuleList(
            [_make_expert(in_dim, hidden_mult, num_classes, dropout_p) for _ in range(self.num_experts)]
        )

    def forward(self, h: torch.Tensor, return_gate: bool = False):
        """
        h: (B, D)
        """
        # --- Gating ---
        # Raw logits for experts
        gate_logits = self.gate(h)  # (B, E)
        # Temperature-scaled softmax for smoother/peaky routing control
        # smaller temperature means more peaky routing (i.e. more confident in the routing decision)
        probs = F.softmax(gate_logits / self.temperature, dim=-1)  # (B, E)

        # --- Experts ---
        # Collect each expert's logits for the batch: (B, E, C)
        #expert(h) = (B, num_classes)
        expert_logits = torch.stack([expert(h) for expert in self.experts], dim=1)

        # --- Mixture ---
        # Weighted average across experts -> final logits: (B, C)
        # (B, E, 1) * (B, E, C) = (B, E, C) -> .sum(dim=1) = (B, C) => weighted average across experts
        logits = (probs.unsqueeze(-1) * expert_logits).sum(dim=1)

        # No selected indices (sel_idx) in soft routing; no aux loss for SoftMoE
        sel_idx = None
        aux_loss = None

        return self.pack(
            logits=logits,
            probs=probs,
            sel_idx=sel_idx,
            aux_loss=aux_loss,
            return_gate=return_gate,
        )