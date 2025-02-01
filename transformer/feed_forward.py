import torch.nn as nn

from torch import Tensor

from .swiglu import swiglu


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            swiglu,
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
