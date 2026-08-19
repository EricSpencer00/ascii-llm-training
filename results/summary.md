# Status summary

One-page snapshot. See `docs/roadmap.md` for phase definitions and
acceptance criteria, `results/bench.md` and `results/legacy_ocr.md` for the
full detail behind the numbers below.

## What exists

| Component | Path | Status |
|---|---|---|
| Deterministic converter (luminance/structure/edge render, rasterize) | `asciiart/render.py`, `asciiart/cli.py` | done (P0) |
| Verifier (SSIM + edge-alignment) | `asciiart/verify.py` | done (P1) |
| Simulated-annealing baseline | `asciiart/anneal.py` | done (P1) |
| Bench harness | `asciiart/bench.py` | done, 20-image run recorded |
| Constrained decoding (logit masking) | `rlvr/constrained.py` | done-baseline (P2) |
| Synthetic RLVR tasks | `rlvr/tasks.py` | done |
| GRPO reward wrapper | `rlvr/reward.py` | done |
| GRPO training entry point | `rlvr/train_grpo.py` | scaffolded, short run in progress on Sophia (P3) |
| `transformer_engine` import-shadow stub | `rlvr/_te_stub/` | done (works around a broken `libtransformer_engine.so` vs. loaded `libcublasLt` on Sophia that crashes `import peft`) |
| PBS job scripts | `scripts/{train_sophia.pbs,rlvr_sophia.pbs,run_baseline_sophia.sh}` | done |
| CI | `.github/workflows/ci.yml` | done -- runs `pytest tests/test_render.py tests/test_verify.py tests/test_anneal.py` on ubuntu-latest, does not run `test_pipeline.py` (needs `torch`+`pyfiglet`) or `test_rlvr.py` (needs `torch`+`transformers`) |
| Legacy word-OCR pipeline, redesigned | `ascii_generator.py`, `data_prep.py`, `model.py`, `train.py`, `evaluate.py` | code done (2D grid positional embeddings + per-char cross-attention query heads); GPU validation run not yet completed as of this writeup |

## Numbers

### P0/P1 -- converter + verifier + anneal (`results/bench.md`)

20 synthetic images, cols=40, anneal_steps=2000. Mean reward (SSIM + edge term):

| method | ssim | edge_score | reward |
|---|---|---|---|
| luminance | 0.040 | 0.162 | 0.089 |
| edge | 0.044 | 0.116 | 0.073 |
| structure | 0.059 | 0.541 | 0.252 |
| anneal (2000 steps) | 0.057 | 0.592 | 0.271 |

Ordering: anneal > structure > luminance > edge, matching the expectation
that search on top of the verifier beats any single deterministic tier.
This is a 20-image informal bench, not the 50-image evaluation set the
roadmap's acceptance criteria reference -- the ordering is directionally
right but the specific 40/50 and 45/50 image-count criteria have not been
checked.

### P2/P3 -- RLVR baseline (Qwen2.5-0.5B-Instruct, 24x12 grid, 20 tasks)

Two eval runs exist in `rlvr/logs/` on Sophia:

- `eval_1787114710.jsonl` (earlier run): constrained and unconstrained both
  report `format_pass_rate: 0.0`, `mean_reward: 0.0`; oracle mean reward
  0.326. An all-zero constrained result looks like a bug or a broken first
  attempt (constrained decoding is a hard structural guarantee -- 0% pass
  rate under masking that forces valid rows is not an expected outcome) and
  should not be treated as the real baseline number.
- `eval_1787114759.jsonl` (later run, the one to cite as the baseline):
  constrained format pass-rate 1.00, mean reward 0.216; unconstrained
  format pass-rate 0.00, mean reward 0.0; oracle (structure converter)
  mean reward 0.295.

Reading: constrained decoding delivers its P2 guarantee (100% valid grids)
and the base model's output does not follow the format at all without it
(0% unconstrained). The policy's constrained mean reward (0.216) is still
well below the deterministic oracle (0.295), which is expected pre-training
and is exactly the gap P3's GRPO run is meant to close.

### GPU jobs (final check 2026-08-19 01:30 CDT)

- **175613** (`ascii-llm-train`, improved OCR, 30k samples, 15 epochs): **finished**.
  Best epoch 14: per_char_acc 0.819, exact_acc 0.545 (final epoch 15: 0.810 / 0.531).
  Baseline job 175601 (same recipe, old model, 10k samples): 0.122 / 0.002.
- **175612** (`ascii-rlvr`, attempt 1) died on tokenizer load — compute nodes have no
  internet; fixed with `HF_HUB_OFFLINE=1`. Attempt 2 (175614) ran the baseline eval on GPU
  and produced `eval_1787118629.jsonl`: constrained format pass-rate 1.00, mean reward
  0.135; unconstrained 0.00 / 0.0; oracle 0.351. It then crashed importing
  `trl.trainer.grpo_trainer` (system-site vllm `.so` has an undefined torch symbol);
  fixed with the `rlvr/_te_stub/vllm` shadow package. Attempt 3 (175616) re-ran the
  baseline eval (`eval_1787120901.jsonl`: constrained 1.00 / 0.143, unconstrained 0 / 0,
  oracle 0.351) then crashed in `GRPOTrainer.__init__` because the `RewardLogger`
  callable had no `__name__`; fixed. Attempt 4 (**175617-ish, submitted 01:52 CDT**) is
  the first one expected to actually reach GRPO steps; check `rlvr/logs/*.jsonl` and
  `logs/rlvr.out` on Sophia for the 50-step curve.

Re-check with:

```bash
ssh sophia "qstat -u eric-spencer; grep -v it/s /grand/EVITA/eric-spencer/ascii-llm-training/logs/rlvr.out | tail; ls /grand/EVITA/eric-spencer/ascii-llm-training/rlvr/logs"
```

### Legacy word-OCR baseline (before redesign)

Job 175601, 10k samples, 15 epochs, d_model=192 (`results/legacy_ocr.md`):

| epoch | train_loss | val_loss | per_char_acc | exact_acc |
|-------|-----------|----------|---------------|-----------|
| 1 | 3.292 | 3.272 | 0.037 | 0.000 |
| 15 | 2.567 | 2.901 | 0.122 | 0.002 |

Diagnosed cause: 1D positional encoding over a 2D grid, plus a
mean-pool-then-single-Linear head that collapses ~1200 mostly-padding
tokens into one vector before predicting all 12 output characters at once.
Redesign (`model.py`, `data_prep.py`): `Grid2DPositionalEmbedding` (learned
row + column embeddings) and per-character learned query vectors that
cross-attend into the full encoder output via `nn.MultiheadAttention`,
instead of mean-pooling. 6 unit tests covering both the legacy and new grid
path pass. GPU validation (job 175613, 30k samples, 15 epochs): **per_char_acc 0.819,
exact_acc 0.545** at the best epoch, versus 0.122 / 0.002 for the old model.

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[test]"

# converter / verifier / anneal
python -m asciiart.cli render path/to/image.png --cols 80 --mode structure
python -m asciiart.cli verify path/to/image.png path/to/output.txt
python -m asciiart.cli anneal path/to/image.png --cols 80 --rows 24
python -m asciiart.bench   # regenerates results/bench.md-style numbers

# legacy OCR pipeline
python ascii_generator.py --num-samples 5000 --out-dir data
python data_prep.py --data-file data/dataset.jsonl --out npz/art_dataset.npz
python train.py --data npz/art_dataset.npz --epochs 15 --d-model 192
python evaluate.py --data npz/art_dataset.npz --checkpoint checkpoints/best.pt --samples 10

# RLVR (on Sophia, GPU)
qsub scripts/rlvr_sophia.pbs
qsub scripts/train_sophia.pbs
```

Local `pytest -q`: with `PYTHONPATH=.`, excluding `tests/test_pipeline.py`
(needs `pyfiglet`, not installed in this checkout's `.venv` and `.venv` has
no `pip` to add it) -- 34 passed, 0 failed. `tests/test_rlvr.py` needs
`torch`/`transformers`, both present in `.venv`, and passes as part of that run.

## Open issues

- `eval_1787114710.jsonl` on sophia shows an all-zero constrained result
  that contradicts the constrained-decoding guarantee (100% valid rows by
  construction) -- looks like a broken first attempt superseded by
  `eval_1787114759.jsonl`, but the root cause was not diagnosed. Worth a
  look before citing this baseline anywhere further.
- P1 bench (`results/bench.md`) used 20 synthetic images at 40 cols, not
  the 50-image evaluation set `docs/roadmap.md`'s P1/P3 acceptance criteria
  are written against. Ordering is directionally consistent but the exact
  image-count criteria (40/50, 45/50) have not been checked.
- P3 acceptance criteria (35/50 images improved over base model,
  no-stratum-regression check, reward-hacking checks from `docs/design.md`)
  have not been run -- only a baseline eval and a short 50-step GRPO smoke
  run are in flight.
- Legacy-OCR redesign has unit-test coverage but no completed GPU training
  run showing per_char_acc/exact_acc above the 0.122/0.002 baseline; job
  175613 is intended to produce that number.
- `tests/test_pipeline.py` cannot run in this checkout's `.venv` (no `pip`
  available to install `pyfiglet`); CI installs it separately and is not
  affected.
- CLI `--help` at the top level (`python -m asciiart.cli --help`) previously
  fell through to the `render` subparser's help instead of listing
  `render`/`verify`/`anneal`; fixed in `asciiart/cli.py`.
