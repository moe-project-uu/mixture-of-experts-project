import pandas as pd
import torch
from typing import Tuple, Dict, Any

# For model statistics
from fvcore.nn import FlopCountAnalysis

def calculate_model_stats(model: torch.nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict[str, float]:
    """
    Calculates and prints the FLOPs and parameter count of a model.
    """
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_shape).to(device)
    
    # Calculate FLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()

    # Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "gflops": total_flops / 1e9,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
    }
    
    print(f"Model Statistics:")
    print(f"  - GFLOPs: {stats['gflops']:.2f}")
    print(f"  - Total Parameters: {stats['total_params_m']:.2f} M")
    print(f"  - Trainable Parameters: {stats['trainable_params_m']:.2f} M")
    
    return stats