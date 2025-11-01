# src/moe/heads/sparse_moe.py
from .base import BaseHead
from moe.utils.losses import shazeer_importance_loss, shazeer_load_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe.utils.experts_and_gating import _make_expert
from moe.utils.experts_and_gating import _NoisyTopKGate



class SparseMoEHead(BaseHead):
    """
    Noisy Top-k Sparse MoE head (Shazeer, 2017).
    Interface:
      forward(h, return_gate=False)
        -> logits (B,C)
      forward(h, return_gate=True)
        -> (logits, probs, sel_idx, aux_loss)
    """
    head_name = "SparseMoE"

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 10,
        num_experts: int = 8,
        hidden_mult: float = 0.0625,
        k: int = 2,
        temperature: float = 1.0,
        dropout_p: float = 0.1,
        gate_input_dropout: float = 0.0,
        gate_logits_dropout: float = 0.0,
        importance_coef: float = 0.1, #hyperparameter for the importance loss
        load_coef: float = 0.1, #hyperparameter for the load loss
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_mult = float(hidden_mult)
        self.k = int(k)
        self.importance_coef = float(importance_coef)
        self.load_coef = float(load_coef)

        # initialize the noisy top-k gate
        self.gate = _NoisyTopKGate(
            in_dim=in_dim,
            num_experts=self.num_experts,
            k=self.k,
            temperature=temperature,
            gate_input_dropout=gate_input_dropout or dropout_p,
            gate_logits_dropout=gate_logits_dropout or dropout_p,
        )

        # initialize the experts (list of experts)
        hidden = int(self.hidden_mult * in_dim) #hidden width of the experts
        self.experts = nn.ModuleList(
            [_make_expert(in_dim, hidden, num_classes, dropout_p) for _ in range(self.num_experts)]
        )

        # for consistency with Dense/Soft MoE heads
        self.expert_width = hidden
        self.total_width = self.num_experts * hidden
        self.capacity_factor = None #yet to be implemented

    def forward(self, h: torch.Tensor, return_gate: bool = True):
        """
        inputs:
        h: (B, D) - input features
        return_gate: bool - whether to return the gate probabilities and topk indices
        output (if return_gate is True, else only logits is returned):
        logits: (B, C) - output logits
        probs: (B, E) - gate probabilities
        sel_idx: (B, k) - topk indices
        aux_loss: (scalar tensor) - auxiliary loss
        return_gate: bool - whether to return the gate probabilities and topk indices
        """
        # --- gating ---
        probs, topk_idx = self.gate(h)   # probs: (B,E) sparse; topk_idx: (B,k)

        # --- Experts (compute-sparse): run only chosen experts and get logits for them only (scatter-add) ---
        B, _ = probs.shape #B is the batch size, _ is the number of experts
        C = self.experts[0].net[-1].out_features  # num_classes
        logits = h.new_zeros(B, C)
        # For each expert, pick only the samples where it's in top-k
        for e, expert in enumerate(self.experts):
            #for each expert, we check if it is in the topk for any of the samples in the batch
            mask = (topk_idx == e).any(dim=1) # (B,) -> True for the samples where the expert is in the topk
            if not mask.any():
                continue
            h_e = h[mask]     # (b_e, D) this is all samples where expert e is in the topk
            out_e = expert(h_e)   # (b_e, C) for of these samples, we get the logits from the expert
            w = probs[mask, e].unsqueeze(1)    # (b_e, 1) this is the probability of the expert being selected for the samples where it is in the topk
            logits[mask] += w * out_e # shape (b_e, C) --- scatter-add into final logits
            #final logits shape should be (B, C)

        # --- Shazeer aux losses: both are w * CV^2 ---
        # importance: total gate mass per expert
        imp_loss = shazeer_importance_loss(probs, self.importance_coef)

        # load: expected #tokens per expert; proxy = how often expert is selected by top-k
        # probs is zero off-topk, >0 on top-k; so count nonzeros per expert in batch
        load_vec = probs.sum(dim=0) # this is the expected number of tokens per expert in the batch (E,)
        load_loss = shazeer_load_loss(load_vec, self.load_coef)

        aux_loss = imp_loss + load_loss
        # if both coefs are 0, make aux_loss None for clean downstream handling
        if (self.importance_coef == 0.0) and (self.load_coef == 0.0):
            aux_loss = None

        return self.pack(
            logits=logits,
            probs=probs,
            sel_idx=topk_idx,
            aux_loss=aux_loss,
            return_gate=return_gate,
        )