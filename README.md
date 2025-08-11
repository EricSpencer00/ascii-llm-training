# ASCII LLM Training

This project generates synthetic ASCII art renderings of random words and trains a small Transformer-based model to recover ("read") the original word from the ASCII art. It demonstrates a lightweight OCR-like pipeline confined entirely to plain text.

## Overview

1. Data Generation (`ascii_generator.py`)
   - Uses `pyfiglet` to render random words with a chosen FIGlet font (default: `standard`).
   - Saves individual samples and a consolidated metadata file (`data/dataset.jsonl`).

2. Dataset Preparation (`data_prep.py`)
   - Builds vocabularies for ASCII-art glyph characters and target word characters.
   - Converts samples into tensor-friendly numpy arrays, storing an npz archive (`art_dataset.npz`) plus vocab JSON.

3. Model (`model.py`)
   - Transformer encoder over the flattened ASCII art token sequence.
   - Multi-head (per position) classifier to predict each character of the underlying word up to `MAX_WORD_LEN`.
   - Uses a PAD / BLANK token for unused positions.

4. Training (`train.py`)
   - Loads prepared dataset, splits train/val.
   - Trains with cross-entropy over each character position.
   - Reports per-position and exact-match accuracy; saves best model.

5. Evaluation (`evaluate.py`)
   - Loads a saved checkpoint; reports accuracy on held-out or newly generated eval set.
   - Can print a few qualitative predictions.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate data (adjust --num-samples as desired)
python ascii_generator.py --num-samples 5000 --out-dir data

# 2. Prepare dataset
python data_prep.py --data-file data/dataset.jsonl --out npz/art_dataset.npz

# 3. Train model
python train.py --data npz/art_dataset.npz --epochs 15 --d-model 192

# 4. Evaluate
python evaluate.py --data npz/art_dataset.npz --checkpoint checkpoints/best.pt --samples 10
```

## Customization
- Change fonts via `--font` in `ascii_generator.py` (run `pyfiglet --list_fonts` to view options).
- Modify `MAX_WORD_LEN` or character sets in `config.py`.
- Increase model depth or width in training arguments.

## File Formats
- `data/dataset.jsonl`: One JSON object per line: `{ "word": "abc", "art": "<multi-line string>", "font": "standard" }`.
- `npz/art_dataset.npz`: Arrays `X` (int64), `Y` (int64), plus metadata like lengths. Input is shape `(N, L)` where `L` is flattened ascii-art token length (padded). Targets shape `(N, MAX_WORD_LEN)`.
- `vocabs.json`: `{ "input_vocab": [...], "target_vocab": [...] }`.

## Notes
- This is a *toy* LLM (transformer encoder) for demonstration. Accuracy depends heavily on word length, font complexity, and dataset size.
- For multi-font generalization, enable `--multi-font` and re-generate dataset.

## Future Ideas
- Sequence-to-sequence decoding with CTC or autoregressive decoder.
- Multi-font / noisy corruption augmentation.
- Export to ONNX for lightweight deployment.

## License
MIT
