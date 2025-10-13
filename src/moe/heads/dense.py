# src/moe/heads/dense.py
from .base import BaseHead
import torch.nn as nn

class DenseHead(BaseHead):
    head_name = "Dense"

    def __init__(self, in_dim, width, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, num_classes),
        )

    def forward(self, h, return_gate=False):
        logits = self.fc(h)
        return self.pack(logits, probs=None, sel_idx=None, aux_loss=None, return_gate=return_gate)
