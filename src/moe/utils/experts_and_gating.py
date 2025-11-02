"""keep all expert and gating-building logic in one place so every MoE variant can reuse the same code instead of duplicating
 expert definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

#####EXPERT BUILDING FUNCTIONS#####
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

#####GATING FUNCTIONS#####
class _NoisyTopKGate(nn.Module):
    """
    Shazeer et al. (2017) noisy top-k gating:
      g = xW_g + ε * softplus(xW_noise)
      keep top-k per row, mask others with -inf, then softmax over experts.
    Returns probs (B,E) and topk indices (B,k).

    Input: (B, D) where D is 512 for CIFAR-10 implementation
    Output: probs and topk_idx
    probs: (B, E) where E is the number of experts -- zero on non-topk, >0 on topk
    topk_idx: (B, k) where k is the number of topk experts -- indices of the topk experts for each sample in the batch
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
        #Normalize gate logits per sample before adding noise (prevents a single huge logit from always winning)
        #Normalization ensures that logits are on a similar scale so that the noise added can actually have an effect on the selection of the topk experts
        #This is key in getting load balancing to work well -- although the Shazeer paper didn't mention this. 
        logits = (logits - logits.mean(dim=-1, keepdim=True)) / (logits.std(dim=-1, keepdim=True) + 1e-5)
        noise_std = F.softplus(self.w_noise(h_in)) + 0.3 # (B, E), strictly positive, this is the learned noise
        noise = torch.randn_like(noise_std) * noise_std # (B, E) output.. element wise multiplication of the learned noise by standard normal noise
        noisy_logits = logits + noise # (B, E) output.. addition of the learned noise to the logits
        # optional dropout on gate logits
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

