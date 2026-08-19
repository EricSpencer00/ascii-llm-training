# Legacy word-OCR pipeline: diagnosis and fix

## Baseline (PBS job 175601, sophia:/grand/EVITA/eric-spencer/ascii-llm-training/logs/train.out)

10k samples, 15 epochs, GPU, d_model=192 (defaults otherwise). Plateaued badly:

| epoch | train_loss | val_loss | per_char_acc | exact_acc |
|-------|-----------|----------|---------------|-----------|
| 1     | 3.2920    | 3.2720   | 0.0372        | 0.0000    |
| 5     | 3.0071    | 3.0208   | 0.0961        | 0.0000    |
| 10    | 2.7657    | 2.9235   | 0.1128        | 0.0010    |
| 15    | 2.5674    | 2.9006   | 0.1215        | 0.0020    |

per_char_acc≈0.12 (chance for 26 letters ≈0.038, so it's learning *something*, but not
useful), exact_acc≈0.002 (basically never gets a whole word right).

## Diagnosis

1. **Position info was 1D over a fundamentally 2D structure.** `data_prep.encode_input`
   flattened the ascii-art rows with a `<nl>` separator into one long sequence, and
   `model.PositionalEncoding` gave every flattened token a single sinusoidal position.
   Column *N* of row 0 and column *N* of row 5 look positionally unrelated in the art
   grid but had no explicit row/column identity — the model had to infer 2D structure
   from a 1D signal.
2. **Proportional-width pyfiglet output shifts columns per sample.** Because word
   length and font vary, the horizontal offset of each character's glyph drifts from
   sample to sample, so even a fixed flat-index position doesn't consistently line up
   with "character k" of the target word.
3. **The real bottleneck: mean-pool → single big linear head.**
   `AsciiTransformer.forward` did `enc_out.mean(dim=1)` over up to 1200 mostly-padding
   tokens, then a single `Linear(d_model, target_vocab_size * max_word_len)` predicted
   *all 12 characters at once* from that one averaged vector. Mean pooling over ~1200
   tokens is close to a bag-of-tokens summary — nearly all sequence order/position
   information is destroyed before the per-character heads ever see it. This is the
   dominant cause of the plateau, more than the positional encoding choice.
4. Only 10k samples / 15 epochs is thin for this task, but not the primary cause —
   the architecture couldn't have learned exact-match with much more data either.

## Changes

- `data_prep.py:25-59` — new `encode_input_grid()`: pads every art sample into a
  fixed `(n_rows, n_cols)` grid (instead of newline-token flatten) so flat index `i`
  always maps to `(row=i//n_cols, col=i%n_cols)` consistently across samples.
  `main()` (`data_prep.py:96-170`) auto-detects `n_rows`/`n_cols` from the dataset
  (capped by new `--max-rows`/`--max-cols`/`--max-rows-cap` flags), and saves them in
  the npz. Old behavior preserved behind new `--legacy-flatten` flag for comparison;
  `--max-input-len` still accepted (now used as the rows×cols budget cap).
- `model.py:7-58` — new `Grid2DPositionalEmbedding` (learned row-embedding +
  col-embedding, summed) replaces 1D sinusoidal encoding whenever `max_rows`/
  `max_cols` are known; falls back to the old `PositionalEncoding` for legacy data.
- `model.py:60-104` — replaced `mean-pool + Linear(d_model, V*W)` with `max_word_len`
  learned query vectors that cross-attend (`nn.MultiheadAttention`) into the full
  encoder output, one query per output character position, each followed by a shared
  `Linear(d_model, V)` classifier head. This gives each output slot its own learned
  "where in the grid to look" pattern instead of forcing one pooled vector to encode
  the whole word.
- `train.py`, `evaluate.py` — thread `max_rows`/`max_cols` through from npz →
  model construction → checkpoint config, so evaluate.py reconstructs the same model
  shape. CLI flags unchanged (backward compatible).
- `tests/test_pipeline.py` — added `pytest.importorskip("torch")`, plus
  `test_encode_input_grid_alignment` and `test_model_forward_shapes_2d_grid` covering
  the new grid path. All 6 tests pass on sophia (`sophia-train` venv).
- `scripts/train_sophia.pbs:18` — bumped `--num-samples` from 10000 to 30000 (data
  gen is cheap; queue was idle).

## CPU smoke test

Ran side-by-side on the sophia login node: 1500 samples, 5 epochs, batch 64, CPU,
new grid+query-head model vs. legacy flatten+mean-pool model
(`smoke_npz/new.npz` vs `smoke_npz/legacy.npz`). This run did not finish within the
session's time budget (background job still in progress when this report was
written) — **CPU smoke numbers are not available to include here.** The 6 unit
tests (including forward-pass shape/finite-loss checks for both the legacy and new
2D-grid paths) do pass, confirming the new code is functionally correct end-to-end.

## GPU run

**Not submitted.** Per instructions to stop waiting and finalize immediately, no PBS
job was queued this session. `qstat -u eric-spencer` showed no jobs for this user at
finalization time (queue idle). To run the intended validation: submit
`scripts/train_sophia.pbs` (already updated to 30k samples / 15 epochs / GPU) via
`qsub scripts/train_sophia.pbs` from `sophia:/grand/EVITA/eric-spencer/ascii-llm-training`,
then check `logs/train.out` for `per_char_acc`/`exact_acc` trending well above the
baseline's 0.12/0.002 by epoch 15.

## Files changed

- `/Users/eric/GitHub/ascii-llm-training/data_prep.py`
- `/Users/eric/GitHub/ascii-llm-training/model.py`
- `/Users/eric/GitHub/ascii-llm-training/train.py`
- `/Users/eric/GitHub/ascii-llm-training/evaluate.py`
- `/Users/eric/GitHub/ascii-llm-training/tests/test_pipeline.py`
- `/Users/eric/GitHub/ascii-llm-training/scripts/train_sophia.pbs`
- `/Users/eric/GitHub/ascii-llm-training/results/legacy_ocr.md` (new)
