import torch
from torch import nn
import torch.nn.functional as F


class SoftmaxGating(nn.Module):
    def __init__(self, input_dim: int, expert_num: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=expert_num)
    
    def forward(self, x):
        return self.linear(x)

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        pass
    
    def forward(self, x):
        # TODO
        pass
    
class MoE_Layer(nn.Module):
    def __init__(self, input_dim: int, expert_num: int = 4, top_k: int = 2):
        self.expert_num = expert_num
        self.top_k = top_k
        
        # set-up router
        self.router = SoftmaxGating(input_dim, self.expert_num)
        
        # set-up 
        self.experts = nn.ModuleList([Expert(...) for i in range(self.expert_num)])
    
    def forward(self, x):
        gating_logits = F.softmax(self.router(x), dim=-1) # last dim is the experts
        top_k_logits, top_k_indices = torch.topk(gating_logits, self.k_top, dim=-1)
        
        for i, expert in enumerate(self.experts):
            # get output from expert
            expert_output = expert(x)
            
            # multiply by gating network's output
            # TODO
            
            # aggregate expert outputs
            # TODO 
        return x