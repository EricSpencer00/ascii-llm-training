import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import AsciiTransformer
from config import PAD_TOKEN


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    return data


class AsciiDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.Y[idx], dtype=torch.long)


def accuracy_metrics(logits, targets, pad_idx):
    # logits: (B, W, V) targets: (B, W)
    pred = logits.argmax(dim=-1)
    mask = targets != pad_idx
    correct = (pred == targets) & mask
    per_char_acc = correct.sum().item() / mask.sum().item()
    exact = ((pred == targets) | ~mask).all(dim=1).float().mean().item()
    return per_char_acc, exact


def main():
    ap = argparse.ArgumentParser(description='Train transformer to read ASCII art words')
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--ff', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--val-split', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_npz(args.data)
    X = d['X']
    Y = d['Y']
    input_vocab = d['input_vocab'].tolist()
    target_vocab = d['target_vocab'].tolist()
    max_input_len = int(d['max_input_len'])
    max_word_len = int(d['max_word_len'])

    pad_idx = target_vocab.index(PAD_TOKEN)

    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(idx, test_size=args.val_split, random_state=args.seed)
    train_ds = AsciiDataset(X[train_idx], Y[train_idx])
    val_ds = AsciiDataset(X[val_idx], Y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = AsciiTransformer(len(input_vocab), len(target_vocab), d_model=args.d_model, nhead=args.nhead,
                             num_layers=args.layers, dim_feedforward=args.ff, max_input_len=max_input_len,
                             max_word_len=max_word_len, dropout=args.dropout)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f'Epoch {epoch} train'):
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            optimizer.zero_grad()
            logits = model(xb)  # (B, W, V)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        all_per_char = []
        all_exact = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f'Epoch {epoch} val'):
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                logits = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                val_loss += loss.item() * xb.size(0)
                per_char, exact = accuracy_metrics(logits, yb, pad_idx)
                all_per_char.append(per_char)
                all_exact.append(exact)
        val_loss /= len(val_ds)
        per_char_acc = sum(all_per_char)/len(all_per_char)
        exact_acc = sum(all_exact)/len(all_exact)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f} per_char_acc={per_char_acc:.4f} exact_acc={exact_acc:.4f}")

        if per_char_acc > best_val:
            best_val = per_char_acc
            ckpt_path = ckpt_dir / 'best.pt'
            torch.save({
                'model_state': model.state_dict(),
                'config': {
                    'input_vocab': input_vocab,
                    'target_vocab': target_vocab,
                    'max_input_len': max_input_len,
                    'max_word_len': max_word_len,
                    'model_args': {
                        'd_model': args.d_model,
                        'nhead': args.nhead,
                        'layers': args.layers,
                        'ff': args.ff,
                        'dropout': args.dropout
                    }
                }
            }, ckpt_path)
            print(f"Saved new best checkpoint to {ckpt_path}")

    print("Training complete.")


if __name__ == '__main__':
    main()
