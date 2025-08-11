import argparse
from pathlib import Path
import numpy as np
import torch

from model import AsciiTransformer
from config import PAD_TOKEN


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    return ckpt


def decode(pred_indices, target_vocab):
    return ''.join(target_vocab[i] for i in pred_indices if target_vocab[i] != PAD_TOKEN)


def main():
    ap = argparse.ArgumentParser(description='Evaluate trained model')
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--samples', type=int, default=5, help='print N random samples')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    input_vocab = d['input_vocab'].tolist()
    target_vocab = d['target_vocab'].tolist()
    max_input_len = int(d['max_input_len'])
    max_word_len = int(d['max_word_len'])
    pad_idx = target_vocab.index(PAD_TOKEN)

    ckpt = load_checkpoint(args.checkpoint)
    model_args = ckpt['config']['model_args']
    model = AsciiTransformer(len(input_vocab), len(target_vocab), max_input_len=max_input_len,
                             max_word_len=max_word_len, **model_args)
    model.load_state_dict(ckpt['model_state'])
    model.to(args.device)
    model.eval()

    # Evaluate overall accuracy
    import random
    idxs = np.arange(len(X))
    random.shuffle(idxs)
    subset = idxs[:min(1000, len(idxs))]

    correct_chars = 0
    total_chars = 0
    exact_matches = 0

    with torch.no_grad():
        for i in subset:
            xb = torch.tensor(X[i:i+1], dtype=torch.long, device=args.device)
            yb = torch.tensor(Y[i:i+1], dtype=torch.long, device=args.device)
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            mask = yb != pad_idx
            correct_chars += ((pred == yb) & mask).sum().item()
            total_chars += mask.sum().item()
            if ((pred == yb) | ~mask).all():
                exact_matches += 1

    print(f"Subset char accuracy: {correct_chars/total_chars:.4f} | exact word accuracy: {exact_matches/len(subset):.4f}")

    # Qualitative samples
    for i in subset[:args.samples]:
        xb = torch.tensor(X[i:i+1], dtype=torch.long, device=args.device)
        logits = model(xb)
        pred = logits.argmax(dim=-1).cpu().numpy()[0]
        true_word = decode(Y[i], target_vocab)
        pred_word = decode(pred, target_vocab)
        print(f"true={true_word} pred={pred_word}")


if __name__ == '__main__':
    main()
