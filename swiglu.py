import torch.nn.functional as F

from torch import Tensor


def swiglu(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * F.silu(x2)
