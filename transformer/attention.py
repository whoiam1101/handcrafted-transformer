import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


NEG_INF = float(-1e10)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor = None
) -> Tensor:
    d = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d)
    if mask is not None:
        scores.masked_fill_(mask == 0, NEG_INF)
    attention_out = F.softmax(scores, dim=-1)
    return torch.matmul(attention_out, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.d = d_model // num_heads
        self.num_heads = num_heads

        self.query_fn = nn.Linear(d_model, d_model)
        self.key_fn = nn.Linear(d_model, d_model)
        self.value_fn = nn.Linear(d_model, d_model)
        self.final_layer = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None
    ) -> Tensor:
        batch_size = query.size(0)

        query = (
            self.query_fn(query)
            .view(batch_size, -1, self.num_heads, self.d)
            .transpose(1, 2)
        )
        key = (
            self.key_fn(key)
            .view(batch_size, -1, self.num_heads, self.d)
            .transpose(1, 2)
        )
        value = (
            self.value_fn(value)
            .view(batch_size, -1, self.num_heads, self.d)
            .transpose(1, 2)
        )

        attention_out = scaled_dot_product_attention(query, key, value, mask)
        attention_out.transpose_(1, 2).contiguous().view_(batch_size, -1, self.d_model)

        return self.final_layer(attention_out)
