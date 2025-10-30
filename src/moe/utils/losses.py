# src/moe/utils/losses.py
import torch
import torch.nn.functional as F

def softmoe_load_balance(probs: torch.Tensor, num_experts: int, eps: float = 1e-8, coef: float = 0.05) -> torch.Tensor:
    """
    Encourages the *mean* routing distribution across the batch to be uniform.

    Inputs:
        probs: (B, E) soft routing probabilities for the batch
        num_experts: E
        eps: numerical clamp for stability
    Output:
        scalar tensor (KL divergence between mean probs and uniform)
    
    Intuition: KL penalizes only the average routing imbalance across
     the batch, not how each sample distributes its probability — so 
     each input can still confidently favor one expert (specialization) 
     while the overall usage remains balanced

    Reasonable coef between 0.01 and 0.1.
    """
    assert probs.dim() == 2, "probs must be (B, E)"
    p = probs.mean(dim=0) #mean over batch --> (E,)
    p = p.clamp_min(eps) #clamp to avoid log(0)
    u = torch.full_like(p, 1.0 / num_experts)      # uniform prior
    # KL(p || u) = sum p * (log p - log u) ; F.kl_div expects inputs: log-probs, targets: probs
    return coef * F.kl_div(p.log(), u, reduction="batchmean")
