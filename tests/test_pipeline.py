"""Tiny CPU smoke tests: generator -> data_prep -> model forward pass."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascii_generator import gen_samples
from data_prep import build_input_vocab, build_target_vocab, encode_input, encode_input_grid, encode_target
from config import MAX_WORD_LEN
from model import AsciiTransformer


def test_gen_samples_basic():
    samples = gen_samples(5, 'standard', [], multi_font=False)
    assert len(samples) == 5
    for s in samples:
        assert s['word']
        assert s['art']
        assert s['font'] == 'standard'


def test_gen_samples_multi_font():
    samples = gen_samples(5, 'standard', [], multi_font=True)
    assert len(samples) == 5
    assert all(s['art'] for s in samples)


def test_data_prep_roundtrip():
    samples = gen_samples(8, 'standard', [], multi_font=False)
    input_vocab = build_input_vocab(samples)
    target_vocab = build_target_vocab()
    in_index = {c: i for i, c in enumerate(input_vocab)}
    tgt_index = {c: i for i, c in enumerate(target_vocab)}

    max_len = 300
    X = np.zeros((len(samples), max_len), dtype=np.int64)
    Y = np.zeros((len(samples), MAX_WORD_LEN), dtype=np.int64)
    for i, s in enumerate(samples):
        X[i] = encode_input(s['art'], in_index, max_len)
        Y[i] = encode_target(s['word'], tgt_index)

    assert X.shape == (8, max_len)
    assert Y.shape == (8, MAX_WORD_LEN)
    assert X.max() < len(input_vocab)
    assert Y.max() < len(target_vocab)


def test_model_forward_shapes():
    samples = gen_samples(4, 'standard', [], multi_font=False)
    input_vocab = build_input_vocab(samples)
    target_vocab = build_target_vocab()
    in_index = {c: i for i, c in enumerate(input_vocab)}
    tgt_index = {c: i for i, c in enumerate(target_vocab)}

    max_len = 300
    X = np.stack([encode_input(s['art'], in_index, max_len) for s in samples])
    Y = np.stack([encode_target(s['word'], tgt_index) for s in samples])

    model = AsciiTransformer(
        input_vocab_size=len(input_vocab),
        target_vocab_size=len(target_vocab),
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        max_input_len=max_len, max_word_len=MAX_WORD_LEN,
    )
    xb = torch.tensor(X, dtype=torch.long)
    logits = model(xb)
    assert logits.shape == (4, MAX_WORD_LEN, len(target_vocab))

    yb = torch.tensor(Y, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), yb.view(-1)
    )
    assert torch.isfinite(loss)


def test_encode_input_grid_alignment():
    samples = gen_samples(6, 'standard', [], multi_font=False)
    input_vocab = build_input_vocab(samples)
    in_index = {c: i for i, c in enumerate(input_vocab)}

    n_rows = max(len(s['art'].split('\n')) for s in samples)
    n_cols = max(len(line) for s in samples for line in s['art'].split('\n'))

    for s in samples:
        grid = encode_input_grid(s['art'], in_index, n_rows, n_cols)
        assert len(grid) == n_rows * n_cols
        assert max(grid) < len(input_vocab)


def test_model_forward_shapes_2d_grid():
    samples = gen_samples(4, 'standard', [], multi_font=False)
    input_vocab = build_input_vocab(samples)
    target_vocab = build_target_vocab()
    in_index = {c: i for i, c in enumerate(input_vocab)}
    tgt_index = {c: i for i, c in enumerate(target_vocab)}

    n_rows = max(len(s['art'].split('\n')) for s in samples)
    n_cols = min(80, max(len(line) for s in samples for line in s['art'].split('\n')))

    X = np.stack([encode_input_grid(s['art'], in_index, n_rows, n_cols) for s in samples])
    Y = np.stack([encode_target(s['word'], tgt_index) for s in samples])

    model = AsciiTransformer(
        input_vocab_size=len(input_vocab),
        target_vocab_size=len(target_vocab),
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        max_input_len=n_rows * n_cols, max_word_len=MAX_WORD_LEN,
        max_rows=n_rows, max_cols=n_cols,
    )
    xb = torch.tensor(X, dtype=torch.long)
    logits = model(xb)
    assert logits.shape == (4, MAX_WORD_LEN, len(target_vocab))

    yb = torch.tensor(Y, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), yb.view(-1)
    )
    assert torch.isfinite(loss)
