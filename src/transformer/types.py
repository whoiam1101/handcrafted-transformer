from torch import Tensor
from typing import Protocol


class AttentionT(Protocol):
    def __init__(self, d_model: int, num_heads: int):          
        pass

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        pass
