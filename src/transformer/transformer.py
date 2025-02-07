import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .attention import MultiHeadAttention
from .encoder import Encoder
from .decoder import Decoder


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
        attention: nn.Module = MultiHeadAttention
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            input_dim,
            max_len,
            dropout,
            attention,
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            output_dim,
            max_len,
            dropout,
            self_attention=attention,
            cross_attention=attention,
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

    @torch.no_grad()
    def generate(
        self,
        sos_token: int,
        eos_token: int,
        temperature: float = 1.0,
        max_len: int = 100,
        src: Tensor | None = None,
        src_mask: Tensor | None = None
    ) -> list[int]:
        """Generate a sequence from the model.

        Args:
            sos_token: Start of sequence token
            eos_token: End of sequence token
            temperature: Sampling temperature (1.0 = greedy)
            max_len: Maximum length of generated sequence
            src: Source sequence (optional)
            src_mask: Source mask (optional)

        Returns:
            List of token indices
        """
        self.eval()

        if src is not None:
            encoder_out = self.encoder(src, src_mask)
        else:
            return []

        generated_seq = [sos_token]
        device = next(self.parameters()).device
        for _ in range(max_len):
            tgt = torch.LongTensor(generated_seq).unsqueeze(0).to(device)
            out = self.decoder(tgt, encoder_out, src_mask)
            probs = F.softmax(out[0, -1] / temperature, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            generated_seq.append(next_token)

            if next_token == eos_token:
                break

        return generated_seq


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

    generate(sos_token, eos_token, temperature=1.0, max_len=100, src=None, src_mask=None):
        Generate a sequence from the model.
        Args:
            sos_token (int): Start-of-sequence token.
            eos_token (int): End-of-sequence token.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            max_len (int, optional): Maximum length of the generated sequence. Defaults to 100.
            src (Tensor, optional): Source input tensor. Defaults to None.
            src_mask (Tensor, optional): Source mask tensor. Defaults to None.
        Returns:
            List[int]: List of generated token indices.
"""
