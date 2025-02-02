import torch
import torch.nn as nn

from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:x.size(1)]


if __name__ == "__main__":
    d_model, max_len = 16, 20
    pe = PositionalEncoding(d_model, max_len)
    
    test_cases = [
        torch.randn(1, 10, d_model),
        torch.randn(32, 15, d_model),
        torch.randn(64, 1, d_model),
    ]
    
    for x in test_cases:
        y = pe(x)
        assert y.shape == x.shape
