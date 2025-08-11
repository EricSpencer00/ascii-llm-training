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


def encode_target(word: str, vocab_index: Dict[str, int]) -> List[int]:
    tokens = [vocab_index[ch] for ch in word[:MAX_WORD_LEN]]
    if len(tokens) < MAX_WORD_LEN:
        tokens.extend([vocab_index[PAD_TOKEN]] * (MAX_WORD_LEN - len(tokens)))
    return tokens


def main():
    ap = argparse.ArgumentParser(description='Prepare dataset npz from jsonl')
    ap.add_argument('--data-file', type=str, required=True)
    ap.add_argument('--out', type=str, default='npz/art_dataset.npz')
    ap.add_argument('--max-input-len', type=int, default=1200, help='truncate/pad length for flattened ascii tokens')
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

    X = np.zeros((len(samples), args.max_input_len), dtype=np.int64)
    Y = np.zeros((len(samples), MAX_WORD_LEN), dtype=np.int64)

    for i, s in tqdm(enumerate(samples), total=len(samples)):
        X[i] = encode_input(s['art'], in_index, args.max_input_len)
        Y[i] = encode_target(s['word'], tgt_index)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, Y=Y, input_vocab=input_vocab, target_vocab=target_vocab,
                        max_input_len=args.max_input_len, max_word_len=MAX_WORD_LEN)

    # Save vocab JSON for readability
    with open(out_path.parent / 'vocabs.json', 'w') as vf:
        json.dump({'input_vocab': input_vocab, 'target_vocab': target_vocab}, vf, indent=2)

    print(f"Saved arrays to {out_path} | X shape {X.shape} | Y shape {Y.shape}")


if __name__ == '__main__':
    main()
