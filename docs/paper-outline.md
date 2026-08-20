# Paper outline

Status: draft outline only. Numbers are deliberately omitted -- every claim
below must be filled from `results/` at write-up time, and several are still
pending (the GRPO re-run against the corrected verifier).

## What this repo actually establishes

Ordered by how well the evidence supports the claim, strongest first.

1. **A plausible perceptual reward for rasterized text is actively
   misleading.** Pixel-level SSIM plus edge-F1 ranks a visibly-correct render
   far *below* a sparse render that ignores the shape. Two mechanisms, each
   independently measurable: windowed SSIM cannot separate a blank grid from
   a perfect render on sparse ink, and pixel edge-F1 reads glyph texture
   inside a correctly-filled region as spurious edges, so filling is punished.
   This is the finding with the cleanest evidence and the widest reach: it
   applies to anyone scoring rendered text or sparse binary raster output.
2. **Scoring at the medium's own resolution fixes it.** Comparing per-cell
   ink coverage against per-cell target brightness restores a sane ranking
   across correct / wrong-shape / shuffled / noise / degenerate candidates.
3. **Degeneracy must be zero by construction, not by penalty.** Blank and
   solid-fill outputs collected a floor from every metric variant tried until
   the metric was built so that a no-variance output cannot score. A
   subtracted-baseline patch fixed one symptom and left the weight dead.
4. **Duplicated reward definitions silently diverge.** The search kept its own
   copy of the reward formula and, after the verifier moved to cell
   resolution, returned output scoring below its own initialization. The
   verifier must be the single definition that both search and RL call.
5. **Constrained decoding makes the format failure class vanish.** Exact
   width/height/charset holds by construction during both eval and RL rollout
   sampling; the unconstrained baseline satisfies it approximately never.
6. **The reward must gate the rollout, not just the scoring.** RL against a
   verifier whose hard predicates are checked only post-hoc produced
   identically-zero reward and zero gradient -- a null result that looks like
   "RL does not work here" and is actually a plumbing bug.
7. **RLVR against the renderer produces a real held-out gain** -- and stays
   far below the deterministic converter it is scored against. (Pending
   re-measurement against the corrected verifier.)

## Framing options

- **A (recommended): a methods/negative-results paper about verifier design.**
  "What makes a renderer a usable verifier for RLVR on grid-structured text."
  ASCII art is the testbed, not the subject. Travels to tables, box drawing,
  circuit diagrams, SVG/TikZ, any render-and-compare reward.
- **B: a benchmark/testbed paper.** Ships the task suite, the verifier, the
  deterministic-converter oracle, and the annealing upper baseline as a
  reference environment for RLVR research.
- **C: an ASCII-art generation paper.** Weakest: the deterministic converter
  beats the learned policy, so the artifact is not the contribution.

## Short paper (4 pages + refs)

Single claim: the reward metric is the experiment.

1. **Intro** -- one paragraph on RLVR needing verifiers; the observation that
   "render it and compare" looks obviously correct and is not; contributions
   as three bullets.
2. **Setup** -- task, hard predicates, constrained decoding, the deterministic
   converter as oracle. Compressed to half a page; details to appendix.
3. **The metric failure** -- the core figure: candidate outputs on one target,
   with each metric's score beside them, showing the inversion. One table of
   the discrimination ladder (correct / wrong-shape / shuffled / noise /
   degenerate) under old and new metrics.
4. **The fix and why it generalizes** -- score at the medium's resolution;
   degeneracy zero by construction; state the property abstractly so it
   transfers off ASCII art.
5. **Consequence for RL** -- short: the ranking inversion propagates into what
   the policy learns; before/after policy samples.
6. **Related work / limitations / conclusion** -- compressed.

Cut for length: the annealing baseline, the legacy OCR model, the full
bench table, all infrastructure debugging.

## Long paper (8 pages + refs)

Adds the environment, the RL results, and the failure taxonomy.

1. **Intro** -- broader claim: verifier design is an under-examined axis of
   RLVR; cheap-and-exact verifiers invite metrics that are neither.
2. **Related work** -- ASCII art generation; RLVR and reward hacking;
   constrained/structured decoding; renderers as reward signal (SVG/TikZ/CAD);
   perceptual metric critiques; 2D structure and tokenization.
3. **Task and environment** -- formal statement of the predicates; the
   generator; the converter oracle; the annealing upper baseline; what makes
   this a good testbed (exact verifier, cheap, visually legible failures).
4. **Verifier design** -- the metric space considered, the inversion result,
   the cell-resolution fix, the degeneracy-by-construction property, and the
   single-definition property. This is the heart.
5. **A taxonomy of failure modes**, each with a reproduction:
   (a) the dead term that contributes nothing but occupies weight;
   (b) the floor that pays degenerate output;
   (c) the metric that inverts ranking;
   (d) the tolerance that pays random noise;
   (e) the reward checked post-hoc instead of gating generation;
   (f) the duplicated reward definition that drifts from the verifier.
6. **RL experiments** -- constrained vs unconstrained format rates; GRPO
   curves with component breakdown (the component split is the anti-hack
   evidence, not the scalar); held-out eval; policy samples; the gap to the
   deterministic converter.
7. **Ablations** -- metric variants; weights; constrained rollout on/off;
   annealing steps as a search-vs-learning comparison.
8. **Discussion** -- when render-and-compare is the right verifier; the
   diagnostic protocol we recommend (score the ladder of degenerate and
   wrong candidates *before* training anything).
9. **Limitations** -- single small model; single font; synthetic targets;
   scale unexplored; the learned policy loses to a deterministic program.

## Figures and tables to build

- F1: the inversion. Correct vs sparse render side by side with both scores.
- F2: discrimination ladder, old metric vs new, as a grouped bar chart.
- F3: GRPO curve with reward decomposed into components (the anti-hack read).
- F4: policy samples, base vs trained vs oracle, at fixed grid size.
- T1: bench across converter modes and the annealing baseline.
- T2: constrained vs unconstrained format pass rate and reward.
- T3: the failure-mode taxonomy with the symptom each produces.

## Reproducibility

MIT-licensed repo, CI, deterministic seeds, one-command bench, PBS scripts for
the RL runs, adapter checkpoint. Every number in the paper should be
regenerable by a script in `results/`.

## Open items before submission

- Re-measure all P3 numbers against the corrected verifier.
- Decide whether the annealing baseline stays (long) or is cut (short).
- The structure-mode under-inking bug is a converter defect, not a metric
  defect -- fix or document it explicitly so the oracle is defensible.
- Consider a second medium (tables or box drawing) to support the
  generalization claim with evidence rather than assertion.
