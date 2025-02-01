import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 input_dim,
                 output_dim,
                 max_len=5000,
                 dropout=0.2):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_dim, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, output_dim, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_out = self.encoder(src, src_mask)
        output = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
        return output

    def predict(self, src, max_len, sos_token=2, eos_token=3):
        self.eval()
        src_mask = None
        encoder_out = self.encoder(src, src_mask)
        preds = []
        tgt = torch.tensor([[sos_token]])

        for _ in range(max_len):
            tgt_mask = None
            output = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
            next_token = output.argmax(-1)[:, -1]
            preds.append(next_token.item())
            if next_token.item() == eos_token:
                break
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
        
        return preds
