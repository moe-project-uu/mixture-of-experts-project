import torch
from torch import nn
import torch.nn.functional as F

class OneLayerExpert(torch.nn.Module):
    """Simple expert with only one hidden layer"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Hidden layer
        self.l_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu_1 = torch.nn.ReLU()
        
        self.output = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu_1(self.l_1(x))
        x = self.output(x)
        return x

class Gating(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, topk=2, device="cpu"):
        super().__init__()
        #here we have a sparse layer trainable weight
        self.l_sparse = torch.nn.Linear(input_size, output_size)
        #here is a noise trainable weight
        self.l_noise = torch.nn.Linear(input_size, output_size)
        self.softplus_noise = torch.nn.Softplus()
        self.topk = topk
        self.device = device

    def forward(self, x):
        #will be of shape batch size x num_experts
        sparsity = self.l_sparse(x)
        noise = self.softplus_noise(self.l_noise(x))
        # print("Noise shape:", noise.shape)
        # print("Sparsity shape:", sparsity.shape)
        # print("noise: ", noise)
        normal_noise  = torch.normal(mean=torch.zeros_like(noise), std=1).to(self.device)
        # print("normal noise shape:", normal_noise.shape)
        # print("normal noise: ", normal_noise)
        noise = normal_noise * noise
        # print("noisy shape:", noise.shape)
        # print("noisy: ", noise)
        x = sparsity + noise
        # print("x shape:", x.shape)  
        # print("x: ", x)
        x = F.softmax(x, dim=-1) 
        return x

class SoftMOE(torch.nn.Module):
    def __init__(self, input_size, hidden_size_experts, hidden_size_gating, num_experts, output_size, device="cpu"):
        super().__init__()
        # Creating gating layer
        self.gate = Gating(
            input_size = input_size, 
            hidden_size = hidden_size_gating, 
            output_size = num_experts,
            device=device
        )
        # Create experts
        self.experts = torch.nn.ModuleList(
            [OneLayerExpert(
                input_size=input_size, 
                hidden_size = hidden_size_experts, 
                output_size = output_size
            ) for _ in range(num_experts)]
        )
        self.to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Get gating outputs
        gating_output = self.gate(x)  # shape: (B, N)
        
        # Batch all expert computations together
        expanded_input = x.unsqueeze(1).expand(-1, len(self.experts), -1)  # (B, N, input_size)
        stacked_input = expanded_input.reshape(-1, expanded_input.shape[-1])  # (B*N, input_size)
        
        # Run all experts in parallel on the batched input
        all_expert_outputs = torch.vstack([
            expert(stacked_input) for expert in self.experts
        ]).reshape(len(self.experts), batch_size, -1)  # (N, B, output_size)
        
        # Transpose to get (B, output_size, N)
        expert_outputs = all_expert_outputs.permute(1, 2, 0)
        
        # Calculate importance loss
        importance = gating_output.sum(0)  # shape: (N,)
        cv = torch.std(importance) / torch.mean(importance)
        loss_importance = 0.1 * cv * cv
        
        # Calculating weighted sum with batch matrix multiplication
        output = torch.bmm(expert_outputs, gating_output.unsqueeze(2)).squeeze(2)
        
        return output, gating_output, loss_importance