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


class Grid2DPositionalEmbedding(nn.Module):
    """Learned row + column positional embedding for a flattened row-major grid.

    The old PositionalEncoding assigned a single 1D sinusoidal position to
    every flattened token, so column N of row 0 and column N of row 3 (which
    are unrelated pixels of the ascii art) looked positionally identical only
    by coincidence of flat index, and characters rendered by pyfiglet at
    proportional widths shifted every later row's columns out of alignment
    with the target word's per-character boundaries. Embedding row and
    column separately (and summing) gives the encoder an actual notion of
    "which scanline, which horizontal offset" that's stable regardless of
    how the row content is padded.
    """

    def __init__(self, d_model: int, max_rows: int, max_cols: int):
        super().__init__()
        self.max_cols = max_cols
        self.row_emb = nn.Embedding(max_rows, d_model)
        self.col_emb = nn.Embedding(max_cols, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) where L == max_rows * max_cols
        L = x.size(1)
        device = x.device
        idx = torch.arange(L, device=device)
        rows = idx // self.max_cols
        cols = idx % self.max_cols
        pos = self.row_emb(rows) + self.col_emb(cols)  # (L, D)
        return x + pos.unsqueeze(0)


class AsciiTransformer(nn.Module):
    def __init__(self, input_vocab_size: int, target_vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, max_input_len: int = 1200, max_word_len: int = 12,
                 dropout: float = 0.1, max_rows: Optional[int] = None, max_cols: Optional[int] = None):
        super().__init__()
        self.token_emb = nn.Embedding(input_vocab_size, d_model)
        self.max_rows = max_rows
        self.max_cols = max_cols
        if max_rows and max_cols:
            # 2D grid input (see data_prep.encode_input_grid): use learned
            # row/col positional embeddings so structure survives flattening.
            self.pos_enc = Grid2DPositionalEmbedding(d_model, max_rows, max_cols)
        else:
            # Legacy 1D sinusoidal positions, kept for backward compatibility
            # with data prepared via --legacy-flatten.
            self.pos_enc = PositionalEncoding(d_model, max_len=max_input_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_word_len = max_word_len

        # Per-position learned query cross-attends into the encoded sequence
        # to predict each output character. This replaces the old
        # mean-pool -> single Linear(d_model, V*W) head, which threw away
        # all positional/order information before the per-character
        # prediction ever saw it (averaging over ~1200 mostly-padding
        # tokens is close to a bag-of-tokens summary) and forced one vector
        # to linearly encode all W characters at once. Queries give each
        # output slot its own learned "where to look" attention pattern.
        self.word_queries = nn.Parameter(torch.randn(max_word_len, d_model) * 0.02)
        self.query_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, target_vocab_size)
        self.target_vocab_size = target_vocab_size
        self.max_input_len = max_input_len

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        # x: (B, L)
        B = x.size(0)
        emb = self.token_emb(x)
        emb = self.pos_enc(emb)
        enc_out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)  # (B, L, D)

        queries = self.word_queries.unsqueeze(0).expand(B, -1, -1)  # (B, W, D)
        attn_out, _ = self.query_attn(queries, enc_out, enc_out, key_padding_mask=src_key_padding_mask)
        attn_out = self.query_norm(attn_out + queries)  # (B, W, D)
        logits = self.classifier(attn_out)  # (B, W, V)
        return logits
