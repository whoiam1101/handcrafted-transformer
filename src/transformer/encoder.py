import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .positional_encoding import PositionalEncoding
from .feed_forward import FeedForward
from .attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.2,
        attention: nn.Module = MultiHeadAttention
    ):
        super().__init__()

        self.attention_fn = attention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        attention = self.attention_fn(x, x, x, mask)
        x = x + self.dropout1(attention)
        x = F.relu(self.norm1(x))

        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = F.relu(self.norm2(x))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_dim: int,
        max_len: int = 5000,
        dropout: float = 0.2,
        attention: nn.Module = MultiHeadAttention
    ):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, attention)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.embedding(src)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
