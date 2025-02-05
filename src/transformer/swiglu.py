import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)
