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
        src_mask: Tensor | None = None,
        top_k: int | None = None,
        top_p: float | None = None
    ) -> list[int]:
        """
        Generate a sequence from the model using autoregressive decoding.

        Args:
            sos_token (int): Start-of-sequence token. The generation starts with this token.
            eos_token (int): End-of-sequence token. The generation stops when this token is produced.
            temperature (float, optional): Sampling temperature. Higher values increase randomness,
                while lower values make the model more deterministic. Default is 1.0.
            max_len (int, optional): Maximum length of the generated sequence. Default is 100.
            src (Tensor, optional): Source input tensor of shape (batch_size, src_seq_len). If provided,
                the encoder output is used as context for the decoder. Default is None.
            src_mask (Tensor, optional): Source mask tensor of shape (batch_size, src_seq_len). If provided,
                it masks out padding tokens in the source sequence. Default is None.
            top_k (int, optional): If set, restricts sampling to the top-k tokens with the highest probabilities.
                Default is None.
            top_p (float, optional): If set, uses nucleus sampling (top-p sampling) with the given cumulative
                probability threshold. Default is None.

        Returns:
            List[int]: A list of generated token indices, including the start-of-sequence token (sos_token)
                and ending with the end-of-sequence token (eos_token) if it is generated within `max_len`.

        Notes:
            - If both `top_k` and `top_p` are provided, `top_k` takes precedence.
            - If neither `top_k` nor `top_p` is provided, the model samples from the full distribution.
        """
        self.eval()

        if src is not None:
            encoder_out = self.encoder(src, src_mask)
        else:
            return []

        generated_seq = [sos_token]
        device = next(self.parameters()).device

        for _ in range(max_len):
            tgt = torch.tensor(generated_seq, dtype=torch.long, device=device).unsqueeze(0)
            out = self.decoder(tgt, encoder_out, src_mask)
            logits = out[0, -1] / temperature
            probs = F.softmax(logits, dim=0)

            next_token = -1
            if top_k is not None:
                top_k_probs, top_k_idxs = torch.topk(probs, top_k)
                top_k_probs /= top_k_probs.sum()
                next_token_idx = torch.multinomial(top_k_probs, 1).item()
                next_token = top_k_idxs[next_token_idx]
            elif top_p is not None:
                sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=True).item()
                truncated_probs = sorted_probs[:cutoff_index + 1]
                truncated_probs /= truncated_probs.sum()
                next_token_idx = torch.multinomial(truncated_probs, 1)
                next_token = sorted_idxs[next_token_idx]
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated_seq.append(int(next_token))

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
    attention (nn.Module, optional): Attention module to use. Default is MultiHeadAttention.

Attributes:
    encoder (Encoder): Encoder part of the transformer.
    decoder (Decoder): Decoder part of the transformer.

Methods:
    forward(src, tgt, src_mask=None, tgt_mask=None):
        Forward pass through the transformer.

        Args:
            src (Tensor): Source input tensor of shape (batch_size, src_seq_len).
            tgt (Tensor): Target input tensor of shape (batch_size, tgt_seq_len).
            src_mask (Tensor, optional): Source mask tensor of shape (batch_size, src_seq_len). Default is None.
            tgt_mask (Tensor, optional): Target mask tensor of shape (batch_size, tgt_seq_len). Default is None.

        Returns:
            Tensor: Output tensor from the decoder of shape (batch_size, tgt_seq_len, output_dim).

    generate(sos_token, eos_token, temperature=1.0, max_len=100, src=None, src_mask=None, top_k=None, top_p=None):
        Generate a sequence from the model using autoregressive decoding.

        Args:
            sos_token (int): Start-of-sequence token.
            eos_token (int): End-of-sequence token.
            temperature (float, optional): Sampling temperature. Higher values increase randomness. Default is 1.0.
            max_len (int, optional): Maximum length of the generated sequence. Default is 100.
            src (Tensor, optional): Source input tensor of shape (batch_size, src_seq_len). Default is None.
            src_mask (Tensor, optional): Source mask tensor of shape (batch_size, src_seq_len). Default is None.
            top_k (int, optional): If set, restricts sampling to the top-k tokens. Default is None.
            top_p (float, optional): If set, uses nucleus sampling with cumulative probability threshold. Default is None.

        Returns:
            List[int]: List of generated token indices.
"""
