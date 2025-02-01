import torch.nn as nn

from swiglu import swiglu


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            swiglu,
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.model(x)
