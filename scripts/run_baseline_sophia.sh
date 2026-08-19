#!/bin/bash
# Small CPU baseline run on the sophia login node, for quick smoke-testing
# the pipeline before submitting a full GPU job via train_sophia.pbs.
set -e
cd /grand/EVITA/eric-spencer/ascii-llm-training

PY=/grand/EVITA/eric-spencer/venvs/sophia-train/bin/python

mkdir -p data npz checkpoints

$PY ascii_generator.py --num-samples 2000 --out-dir data --multi-font
$PY data_prep.py --data-file data/dataset.jsonl --out npz/art_dataset.npz
$PY train.py --data npz/art_dataset.npz --epochs 3 --d-model 192 --device cpu
