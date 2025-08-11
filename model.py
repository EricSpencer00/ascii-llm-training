import math
from typing import Optional
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return x


class AsciiTransformer(nn.Module):
    def __init__(self, input_vocab_size: int, target_vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, max_input_len: int = 1200, max_word_len: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(input_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_input_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_word_len = max_word_len
        self.classifier = nn.Linear(d_model, target_vocab_size * max_word_len)
        self.target_vocab_size = target_vocab_size
        self.max_input_len = max_input_len

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        # x: (B, L)
        emb = self.token_emb(x)
        emb = self.pos_enc(emb)
        enc_out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        # Pool: mean pooling
        pooled = enc_out.mean(dim=1)  # (B, d_model)
        logits = self.classifier(pooled)  # (B, target_vocab_size * max_word_len)
        logits = logits.view(-1, self.max_word_len, self.target_vocab_size)  # (B, W, V)
        return logits
