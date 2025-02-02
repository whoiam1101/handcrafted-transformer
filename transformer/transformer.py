import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .encoder import Encoder
from .decoder import Decoder


__all__ = ["Transformer"]


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_dim: int,
        output_dim: int,
        max_len: int = 5000,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            input_dim,
            max_len,
            dropout
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            output_dim,
            max_len,
            dropout
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None
    ) -> Tensor:
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
        return decoder_out


Transformer.__doc__ = """
Transformer model for sequence-to-sequence tasks.

Args:
    num_layers (int): Number of layers in the encoder and decoder.
    d_model (int): Dimensionality of the model.
    num_heads (int): Number of attention heads.
    d_ff (int): Dimensionality of the feed-forward network.
    input_dim (int): Dimensionality of the input.
    output_dim (int): Dimensionality of the output.
    max_len (int, optional): Maximum length of the input sequences. Default is 5000.
    dropout (float, optional): Dropout rate. Default is 0.2.

Attributes:
    encoder (Encoder): Encoder part of the transformer.
    decoder (Decoder): Decoder part of the transformer.

Methods:
    forward(src, tgt, src_mask=None, tgt_mask=None):
        Forward pass through the transformer.
        Args:
            src (Tensor): Source input tensor.
            tgt (Tensor): Target input tensor.
            src_mask (Tensor, optional): Source mask tensor. Default is None.
            tgt_mask (Tensor, optional): Target mask tensor. Default is None.
        Returns:
            Tensor: Output tensor from the decoder.

    predict(src, max_len, sos_token=2, eos_token=3):
        Generate predictions for the given source input.
        Args:
            src (Tensor): Source input tensor.
            max_len (int): Maximum length of the generated sequence.
            sos_token (int, optional): Start-of-sequence token. Default is 2.
            eos_token (int, optional): End-of-sequence token. Default is 3.
        Returns:
            List[int]: List of predicted token indices.
"""
