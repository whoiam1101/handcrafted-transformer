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
    d = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d)
    if mask is not None:
        scores.masked_fill_(mask == 0, float("-inf"))
    attention_out = F.softmax(scores, dim=-1)
    return torch.matmul(attention_out, value)


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
