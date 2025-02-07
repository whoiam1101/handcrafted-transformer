import torch.nn as nn

from torch import Tensor

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.2,
        self_attention: nn.Module = MultiHeadAttention,
        cross_attention: nn.Module = MultiHeadAttention
    ):
        super().__init__()

        self.self_attention = self_attention(d_model, num_heads)
        self.cross_attention = cross_attention(d_model, num_heads)

        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        attention_out = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout1(attention_out)
        x = self.norm1(x)

        attention_out = self.cross_attention(x, encoder_out, encoder_out, src_mask)
        x = x + self.dropout2(attention_out)
        x = self.norm2(x)

        ff_out = self.ff(x)
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        output_dim: int,
        max_len: int = 5000,
        dropout: float = 0.2,
        self_attention: nn.Module = MultiHeadAttention,
        cross_attention: nn.Module = MultiHeadAttention
    ):
        super().__init__()

        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout,
                self_attention,
                cross_attention
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.final_layer = nn.Linear(d_model, output_dim)

    def forward(
        self,
        tgt: Tensor,
        encoder_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        x = self.norm(x)
        return self.final_layer(x)
