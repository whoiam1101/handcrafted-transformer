import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward
from positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        attention = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout1(attention)
        x = self.norm1(x)

        attention = self.cross_attention(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x)

        ff_out = self.ff(x)
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, output_dim, max_len=5000, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNomr(d_model)
        self.final_layer = nn.Linear(d_model)

    def forward(self, tgt, encoder_out, src_mask, tgt_mask):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        x = self.norm(x)
        return self.final_layer(x)
