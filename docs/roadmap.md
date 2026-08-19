# Roadmap

Phased build order for the converter/verifier/anneal testbed. Each phase has
an acceptance criterion that gates moving to the next; a phase is not "done"
because code exists for it, it's done when its criterion is met on the fixed
50-image evaluation set described in `docs/design.md`.

## P0 — Deterministic converter

Build `asciiart/render.py`: all three rendering tiers (luminance, structure,
edge), `rasterize()` as the inverse primitive, and the CLI (`python -m
asciiart.cli render`).

**Acceptance criteria:**
- All three modes run end-to-end on the 50-image set without crashing.
- Structure-aware mode visibly outperforms luminance-only on the flat/simple
  and text/line-art strata by inspection (side-by-side comparison, not yet a
  metric — the metric doesn't exist until P1).
- Glyph coverage for the luminance ramp is measured per-font, not hardcoded,
  and regenerating it for a second font changes the ramp ordering.

## P1 — Verifier and annealing baseline

Build `asciiart/verify.py` (per-cell SSIM + edge-alignment term) and
`asciiart/anneal.py` (simulated annealing using the verifier as energy).

**Acceptance criteria:**
- Verifier score for `mode="structure"` output against its own source image
  is higher than for `mode="luminance"` output against the same image, on
  at least 40 of the 50 evaluation images — confirms the verifier tracks the
  ordering a human would expect between tiers.
- A uniform-fill grid (all cells the same mid-density glyph) scores below
  the luminance-tier baseline on the low-edge-content stratum — confirms the
  edge-alignment term is actually suppressing the fill degenerate, not just
  present in the formula.
- Annealing converges (energy plateau, not still improving at move budget
  exhaustion) on all 50 images within a fixed move budget, and its output
  beats `mode="luminance"` on the aggregate reward on at least 45 of 50.

## P2 — Constrained decoding

Logit masking for line-length and charset constraints, wired into a
generation loop over an open model (exact model TBD by P2, likely something
already available on Sophia).

**Acceptance criteria:**
- Zero constraint violations (ragged rows, out-of-charset glyphs) on 50/50
  outputs — this is a hard structural guarantee, not a rate to improve, so
  the bar is 100%, not "better than before."
- Generation latency overhead from masking stays under 2x unconstrained
  decoding latency, measured on the same hardware.

## P3 — RLVR / GRPO on a small open model, on Sophia

Train against the P1 verifier as reward, using constrained decoding from P2
so the policy never has to learn width/charset constraints itself and can
spend capacity on structure instead.

**Acceptance criteria:**
- Trained policy's mean SSIM + edge-F1 (against the `mode="structure"`
  reference, per the evaluation protocol) improves over the base model's
  zero-shot attempt on at least 35 of 50 images.
- No stratum regresses below the base model's score by more than a small
  margin — a global average improvement that hides a collapse on one
  stratum (most likely the low-edge-content one, per the reward-hacking
  table in `docs/design.md`) does not count as passing.
- Reward-hacking checks from `docs/design.md` (charset entropy, cross-font
  generalization) run at the end of training and are reported, not just
  the raw reward curve.

## P4 — Writeup

Research note on "rasterize-and-compare as a verifier for RLVR on
grid-structured discrete outputs," using the P0–P3 results as evidence.
Explicit discussion of how the approach generalizes to other grid-structured
discrete outputs with checkable rendering (tables, box-drawing diagrams,
circuit schematics) and where the analogy breaks down (e.g. outputs where
"rasterize" isn't a well-defined or cheap operation).

**Acceptance criteria:**
- Draft covers P0–P3 results with the per-stratum numbers, not aggregates.
- At least one of the generalization targets (tables, box-drawing, circuit
  diagrams) is prototyped far enough to report a concrete number, not just
  asserted as plausible.
