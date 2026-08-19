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

### RLVR / GRPO on Sophia (job 175639, final)

The first four attempts never reached a training step (offline HF cache,
transformers-5 `apply_chat_template` return type, broken system `vllm`/
`transformer_engine` shared objects, TRL requiring `__name__` on reward
callables). Job 175620 then ran 50 steps with **reward identically 0.0 and
grad_norm 0.0**: TRL samples rollouts through its own `generate` call and
exposes no logits-processor hook, so every rollout was unconstrained, every
completion failed the hard predicates, and the reward was zero by
construction. `_patch_generate_with_constraints` in `rlvr/train_grpo.py`
wraps the model's bound `generate` to fix this.

Job 175639, 50 steps, Qwen2.5-0.5B-Instruct + LoRA, 24x12 grid, 8 generations:

| window | mean_reward | mean_ssim | mean_edge | format_pass_rate |
|--------|-------------|-----------|-----------|------------------|
| first 17 steps | 0.1427 | 0.2234 | 0.0216 | 1.000 |
| last 17 steps  | 0.1419 | 0.2244 | 0.0180 | 1.000 |

Baseline eval from the same job: constrained format pass-rate 1.00 / reward
0.131, unconstrained 0.00 / 0.0, converter oracle 0.346.

Reading, stated plainly: constrained decoding works and now holds during
training (100% valid grids on every step, versus 0% unconstrained). GRPO
itself produced **no measurable improvement over 50 steps** -- reward is flat
within noise. That is not evidence the approach fails; 50 steps with a LoRA
on a 0.5B model is a smoke test, not a training run. What it does establish
is that the pipeline is now capable of learning (nonzero, varying reward with
real gradients) where before it structurally could not.

Also worth noting: `mean_edge` is near zero (~0.02) while SSIM carries almost
all of the reward. The policy is producing plausible density but essentially
no edge alignment, so the anti-uniform-fill term is not yet biting on policy
output the way it does on the converter comparison. A longer run should watch
whether reward climbs via SSIM alone -- that would be the uniform-fill hack
appearing, and is the first thing to check before believing any reward gain.


### GRPO learns once the hack is closed (job 175665, 600 steps)

600 steps, Qwen2.5-0.5B-Instruct + LoRA, 24x12 grid, 8 generations, 40 tasks,
2h04m on one GPU. Means over consecutive 100-step windows:

| steps | reward | ssim | edge | format pass |
|-------|--------|------|------|-------------|
| 0-100 | 0.0092 | 0.2283 | 0.0229 | 1.00 |
| 100-200 | 0.0137 | 0.2155 | 0.0342 | 1.00 |
| 200-300 | 0.0249 | 0.2018 | 0.0620 | 1.00 |
| 300-400 | 0.0356 | 0.1815 | 0.0887 | 1.00 |
| 400-500 | 0.0383 | 0.1776 | 0.0954 | 1.00 |
| 500-600 | 0.0402 | 0.1622 | 0.1002 | 1.00 |

Reward rose 4.4x and it is monotonic across all six windows. The direction of
the components is what makes this believable rather than another hack: the
**edge score rose 4.4x while raw SSIM fell** (0.228 -> 0.162). A policy gaming
the density floor would show the opposite -- SSIM climbing, edges flat. Format
pass-rate stays pinned at 1.00 by construction, so none of the gain comes from
learning the grid format; it is all likeness.

Caveats, stated plainly: 0.040 is still an order of magnitude below the
deterministic converter oracle (~0.30), so this is a learning signal, not a
competitive method. The reward is training-time on the training tasks; run
175665 set `save_strategy="no"` and saved no adapter, so there was no artifact
to sample or to score on held-out tasks. Job **175759** repeats the run with
the adapter persisted and a post-training eval on a disjoint task seed, which
is the number that would actually support a claim.

### The empty-grid reward hack (found 2026-08-19, fixed)

Sampling the trained-baseline policy under constrained decoding on a
"filled circle" task returned **twelve rows of spaces** -- a completely blank
grid -- and the verifier paid it **0.190**, against an oracle of 0.393. Blank
output was worth 48% of the converter's score. Raw SSIM is the culprit: the
targets are mostly dark, so drawing nothing matches most of the image. On a
filled-circle target the oracle's raw SSIM (0.493) is actually *below* the
blank grid's (0.494); every bit of real signal lives in the edge term.

Fix: `score()` now reports `ssim_gain = (ssim - ssim_blank) / (1 - ssim_blank)`
clipped at zero, and the reward uses the gain rather than raw SSIM. The blank
grid now scores exactly 0.000. The normalization anchor (blank-vs-target)
differs from the objective, so the policy cannot move it.

This also explains the flat 50-step GRPO curve: reward ~0.14 with `mean_edge`
~0.02 is the signature of a policy sitting on near-blank output, collecting
the SSIM floor and never being pushed off it.

Bench numbers after the fix (previous values in parentheses):

| method | ssim | edge | reward |
|---|---|---|---|
| anneal (2000 steps) | 0.057 | 0.592 | **0.238** (0.271) |
| structure | 0.059 | 0.541 | 0.217 (0.252) |
| luminance | 0.040 | 0.162 | 0.066 (0.089) |
| edge | 0.044 | 0.116 | 0.048 (0.073) |

Ordering is unchanged; every method loses the free SSIM floor it was being
paid before.

### Open issues

- `render(mode="structure")` renders a smooth 0->255 gradient as an
  **all-blank grid**. The per-cell glyph match is dominated by gradient
  structure, so a smooth cell picks space regardless of its brightness. This
  made `test_score_perfect_self_match` vacuous (it asserted a high score on
  blank-vs-blank); the test now uses a circle. The converter behavior itself
  is unfixed.
- The SSIM term contributes almost nothing on these synthetic targets; the
  reward is effectively edge-F1 with an SSIM tiebreaker. A perceptual metric
  better suited to sparse ink would be the real fix.

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
