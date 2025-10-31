# src/moe/heads/soft_moe.py
from .base import BaseHead
from moe.utils.losses import shazeer_importance_loss, shazeer_load_loss
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

class _NoisyTopKGate(nn.Module):
    """
    Shazeer et al. (2017) noisy top-k gating:
      g = xW_g + ε * softplus(xW_noise)
      keep top-k per row, mask others with -inf, then softmax over experts.
    Returns probs (B,E) and topk indices (B,k).

    Input: (B, D) where D is 512 for CIFAR-10 implementation
    Output: probs and topk_idx
    probs: (B, E) where E is the number of experts
    topk_idx: (B, k) where k is the number of topk experts
    """
    def __init__(self, in_dim: int, num_experts: int, k: int = 2, temperature: float = 1.0,
                 gate_input_dropout: float = 0.0, gate_logits_dropout: float = 0.0):
        super().__init__()
        assert k >= 1 and k <= num_experts
        assert temperature > 0.0
        self.k = k
        self.temperature = float(temperature)
        self.w_gate  = nn.Linear(in_dim, num_experts, bias=True)
        self.w_noise = nn.Linear(in_dim, num_experts, bias=True)
        self.gate_in_drop   = nn.Dropout(p=gate_input_dropout) if gate_input_dropout > 0 else None
        self.gate_logits_drop = nn.Dropout(p=gate_logits_dropout) if gate_logits_dropout > 0 else None   

    def forward(self, h: torch.Tensor): #h is the input features (B, D) where D is 512 for CIFAR-10 implementation
        # optional dropout on gate input
        h_in = self.gate_in_drop(h) if self.gate_in_drop is not None else h

        logits = self.w_gate(h_in) # (B, E)
        noise_std = F.softplus(self.w_noise(h_in)) + 1e-1  # (B, E), strictly positive, this is the learned noise
        noise = torch.randn_like(noise_std) * noise_std # (B, E) output.. element wise multiplication of the learned noise by standard normal noise
        noisy_logits = logits + noise # (B, E) output.. addition of the learned noise to the logits

        if self.gate_logits_drop is not None:
            noisy_logits = self.gate_logits_drop(noisy_logits)

        # top-k mask before softmax
        topk_vals, topk_idx = torch.topk(noisy_logits, k=self.k, dim=-1)  # (B,k), (B,k)
        """EX: 
        topk_vals = [[3.4, 2.1]]
        topk_idx = [[1, 3]]
        """
        
        thresh = topk_vals[:, -1:].clone() # (B,1) -- we get the last value of the topk_vals which is the smallest topk logit (we use the smallest logit as threshold to mask the non-topk logits)
        masked = torch.where(noisy_logits >= thresh, noisy_logits, torch.full_like(noisy_logits, float("-inf"))) # (B, E) this is the masked logits
        probs = F.softmax(masked / self.temperature, dim=-1) #softmax over the experts.. (B, E), zero on non-topk   
        
        return probs, topk_idx


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
        probs, topk_idx = self.gate(h)                               # probs: (B,E) sparse; topk_idx: (B,k)

        # --- experts (run all; small E on CIFAR keeps it simple/fast) ---
        expert_logits = torch.stack([e(h) for e in self.experts], dim=1)  # (B,E,C)
        #e(h) is (B, C) for each expert
        #we stack the logits for all the experts to get a tensor of shape (B, E, C)
        # ex output: tensor([[
        #  [ 1,  2,  3],    <- batch 0, expert 0
#          [ 7,  8,  9]],   <- batch 0, expert 1
#
#         [[ 4,  5,  6],    <- batch 1, expert 0
#          [10, 11, 12],    <- batch 1, expert 1
#           ]])  

        # --- combine (sum over experts with experts weighted by their sparse probs) ---
        logits = (probs.unsqueeze(-1) * expert_logits).sum(dim=1)         # (B,C)

        # --- Shazeer aux losses: both are w * CV^2 ---
        # importance: total gate mass per expert
        imp_loss = shazeer_importance_loss(probs, self.importance_coef)

        # load: expected #tokens per expert; proxy = how often expert is selected by top-k
        # probs is zero off-topk, >0 on top-k; so count nonzeros per expert in batch
        load_vec = (probs > 0).float().sum(dim=0) # this is the expected number of tokens per expert in the batch (E,)
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