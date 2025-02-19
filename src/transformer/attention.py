import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from torch import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None
) -> Tensor:
    d_model = query.size(-1)
    scores = query @ key.transpose(-2, -1) / sqrt(d_model)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_out = F.softmax(scores, dim=-1)
    return attention_out @ value


def sparse_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    window_size: int,
    mask: Tensor | None = None
) -> Tensor:
    seq_len, d_model = query.shape[-2:]
    output = torch.zeros_like(query)
    
    for idx in range(seq_len):
        start = max(0, idx - window_size)
        end = min(seq_len, idx + window_size + 1)
        
        Q = query[:, idx:idx+1, :]
        K = key[:, start:end, :]
        V = value[:, start:end, :]

        scores = Q @ K.transpose(-2, -1) / sqrt(d_model)
        if mask is not None:
            local_mask = mask[:, idx:idx+1, start:end]
            scores = scores.masked_fill(local_mask == 0, float("-inf"))
        attention_out = F.softmax(scores, dim=-1)
        output[:, idx:idx+1, :] = attention_out @ V
    
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.query_fn = nn.Linear(d_model, d_model, bias=False)
        self.key_fn = nn.Linear(d_model, d_model, bias=False)
        self.value_fn = nn.Linear(d_model, d_model, bias=False)

        self.final_layer = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        batch_size = query.size(0)

        Q = self.query_fn(query)
        K = self.key_fn(key)
        V = self.value_fn(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        attention_out = scaled_dot_product_attention(Q, K, V, mask)
        attention_out = attention_out.transpose(1, 2).contiguous()
        attention_out = attention_out.view(batch_size, -1, self.d_model)

        return self.final_layer(attention_out)
