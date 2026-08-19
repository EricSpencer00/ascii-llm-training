import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from config import NEWLINE_TOKEN, PAD_INPUT_TOKEN, PAD_TOKEN, TARGET_CHARS, MAX_WORD_LEN


def build_input_vocab(samples: List[Dict]) -> List[str]:
    chars = set()
    for s in samples:
        for line in s['art'].split('\n'):
            for ch in line:
                chars.add(ch)
    vocab = [PAD_INPUT_TOKEN, NEWLINE_TOKEN] + sorted(chars)
    return vocab


def build_target_vocab() -> List[str]:
    return [PAD_TOKEN] + TARGET_CHARS


def encode_input(art: str, vocab_index: Dict[str, int], max_len: int) -> List[int]:
    tokens = []
    for line in art.split('\n'):
        for ch in line:
            tokens.append(vocab_index[ch])
        tokens.append(vocab_index[NEWLINE_TOKEN])
    if len(tokens) < max_len:
        tokens.extend([vocab_index[PAD_INPUT_TOKEN]] * (max_len - len(tokens)))
    else:
        tokens = tokens[:max_len]
    return tokens


def encode_input_grid(art: str, vocab_index: Dict[str, int], n_rows: int, n_cols: int) -> List[int]:
    """Encode ascii art as a fixed (n_rows x n_cols) grid, row-major flattened.

    Unlike encode_input (which appends a newline token and lets the flattened
    sequence run on regardless of row boundaries), this pads/truncates every
    row to exactly n_cols and every sample to exactly n_rows, so that flat
    index i always maps to a consistent (row=i//n_cols, col=i%n_cols). This
    lets the model use real 2D positional embeddings and keeps character
    columns aligned across samples instead of drifting with word length/font.
    """
    pad_id = vocab_index[PAD_INPUT_TOKEN]
    lines = art.split('\n')[:n_rows]
    grid = []
    for line in lines:
        row = [vocab_index.get(ch, pad_id) for ch in line[:n_cols]]
        if len(row) < n_cols:
            row.extend([pad_id] * (n_cols - len(row)))
        grid.append(row)
    while len(grid) < n_rows:
        grid.append([pad_id] * n_cols)
    tokens = [tok for row in grid for tok in row]
    return tokens


def encode_target(word: str, vocab_index: Dict[str, int]) -> List[int]:
    tokens = [vocab_index[ch] for ch in word[:MAX_WORD_LEN]]
    if len(tokens) < MAX_WORD_LEN:
        tokens.extend([vocab_index[PAD_TOKEN]] * (MAX_WORD_LEN - len(tokens)))
    return tokens


def main():
    ap = argparse.ArgumentParser(description='Prepare dataset npz from jsonl')
    ap.add_argument('--data-file', type=str, required=True)
    ap.add_argument('--out', type=str, default='npz/art_dataset.npz')
    ap.add_argument('--max-input-len', type=int, default=1200,
                     help='(legacy) truncate/pad length for flattened ascii tokens; used with --legacy-flatten, '
                          'or as an overall cap (max_rows*max_cols) when auto-sizing the 2D grid')
    ap.add_argument('--max-rows', type=int, default=0,
                     help='rows in the 2D ascii grid encoding (0 = auto-detect from data, capped by --max-rows-cap)')
    ap.add_argument('--max-cols', type=int, default=0,
                     help='cols in the 2D ascii grid encoding (0 = auto-detect from data, capped so rows*cols <= --max-input-len)')
    ap.add_argument('--max-rows-cap', type=int, default=12, help='cap on auto-detected rows')
    ap.add_argument('--legacy-flatten', action='store_true',
                     help='use the old newline-token flatten encoding (destroys row/col alignment; kept for comparison only)')
    args = ap.parse_args()

    data_path = Path(args.data_file)
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line))

    input_vocab = build_input_vocab(samples)
    target_vocab = build_target_vocab()
    in_index = {c: i for i, c in enumerate(input_vocab)}
    tgt_index = {c: i for i, c in enumerate(target_vocab)}

    Y = np.zeros((len(samples), MAX_WORD_LEN), dtype=np.int64)

    if args.legacy_flatten:
        X = np.zeros((len(samples), args.max_input_len), dtype=np.int64)
        for i, s in tqdm(enumerate(samples), total=len(samples)):
            X[i] = encode_input(s['art'], in_index, args.max_input_len)
            Y[i] = encode_target(s['word'], tgt_index)
        max_rows = None
        max_cols = None
        max_input_len = args.max_input_len
    else:
        # Auto-detect a fixed grid size from the data so every row/col index
        # means the same thing across samples (see encode_input_grid).
        data_rows = max(len(s['art'].split('\n')) for s in samples)
        data_cols = max((len(line) for s in samples for line in s['art'].split('\n')), default=1)
        max_rows = args.max_rows or min(data_rows, args.max_rows_cap)
        if args.max_cols:
            max_cols = args.max_cols
        else:
            cap_cols = max(1, args.max_input_len // max_rows)
            max_cols = min(data_cols, cap_cols)
        max_input_len = max_rows * max_cols

        X = np.zeros((len(samples), max_input_len), dtype=np.int64)
        for i, s in tqdm(enumerate(samples), total=len(samples)):
            X[i] = encode_input_grid(s['art'], in_index, max_rows, max_cols)
            Y[i] = encode_target(s['word'], tgt_index)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(X=X, Y=Y, input_vocab=input_vocab, target_vocab=target_vocab,
                        max_input_len=max_input_len, max_word_len=MAX_WORD_LEN)
    if max_rows is not None:
        save_kwargs['max_rows'] = max_rows
        save_kwargs['max_cols'] = max_cols
    np.savez_compressed(out_path, **save_kwargs)

    # Save vocab JSON for readability
    with open(out_path.parent / 'vocabs.json', 'w') as vf:
        json.dump({'input_vocab': input_vocab, 'target_vocab': target_vocab}, vf, indent=2)

    print(f"Saved arrays to {out_path} | X shape {X.shape} | Y shape {Y.shape}")


if __name__ == '__main__':
    main()
