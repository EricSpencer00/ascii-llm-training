# Publication venues for the RLVR / ASCII-art verifier paper

Paper: RLVR with a non-differentiable renderer as verifier, constrained decoding for exact
grid format, ASCII art testbed, negative result on perceptual reward-metric ranking.
Small-scale (Qwen2.5-0.5B, single GPU, 600-step GRPO). Short-paper-sized empirical +
methodological contribution.

Today: 2026-08-19. All deadlines AoE (UTC-12) unless noted. Sources linked inline;
anything not confirmed on a primary/official page is marked UNVERIFIED.

## 1. ACL-family (ARR-gated)

| Venue | Deadline | Commit-to date | Pages | Archival | Accept rate | Fit |
|---|---|---|---|---|---|---|
| ARR Aug 2026 cycle → EACL 2027 | **Submission Aug 3, 2026 — already passed** | Oct 11, 2026 commit | 8 (long)/4 (short) | Yes | ~20-25% (typical ARR) | N/A — deadline missed |
| ARR Oct 2026 cycle → NAACL 2027 / COLING 2027 | **Submission Oct 12, 2026** | Dec 20, 2026 commit | 8 (long)/4 (short) | Yes | ~20-25% | Medium — reachable, but constrained-decoding+RLVR negative result is thin for a full 8-page long paper; a short paper (4pp) fits well |
| EACL 2027 (Athens, Mar 9-14 2027) | Aug 3, 2026 (via ARR Aug cycle) — passed | — | 8/4 | Yes | ~20% | N/A — missed |
| NAACL 2027 (San Francisco, Jun 1-5 2027) | Oct 12, 2026 (via ARR Oct cycle) | — | 8/4 | Yes | ~20-25% | Medium-strong for short paper |
| COLING 2027 (Macau, May 9-14 2027) | Oct 12, 2026 (via ARR Oct cycle) | — | 8/4 | Yes | higher than ACL/EMNLP historically (~30-35%) | Medium — COLING favors resource/method papers, decent fit |
| ACL 2027 | UNVERIFIED — historically a Jan/Feb ARR cycle commits; no page found with exact 2026-cycle date yet | — | 8/4 | Yes | ~20% | Too far out to plan around yet |

**How ARR works if submitting now:** ARR (aclrollingreview.org) runs ~5 review cycles/year
on a 10-week timeline. You submit once to ARR, get reviews + meta-review, then "commit" the
already-reviewed paper to one eligible downstream conference in that cycle's commitment
window. The next cycle you can still make is **October 2026** (submit Oct 12, 2026; commit
by Dec 20, 2026), which feeds **NAACL 2027 or COLING 2027**. The August 2026 cycle (which
fed EACL 2027) closed Aug 3, 2026, before today.
[ARR dates](https://aclrollingreview.org/dates), [EACL 2027 CFP](https://2027.eacl.org/calls/papers/)

## 2. ML venues

| Venue | Deadline | Archival | Accept rate | Fit |
|---|---|---|---|---|
| NeurIPS 2026 Evaluations & Datasets (formerly D&B) track | Abstract May 4 / full paper **May 6, 2026 — passed** | Yes | ~30% (D&B historically higher than main) | N/A this year; note for 2027 — track renamed "Evaluations & Datasets," explicit focus on eval-as-object-of-study, would be a strong future fit for the reward-metric negative result |
| NeurIPS 2026 main track | Passed (spring 2026) | Yes | ~25% | N/A — missed |
| ICLR 2027 | Abstract **Sep 19, 2026**; full paper **Sep 25, 2026** | Yes | ~32% | Medium — ICLR likes RL/decoding methods papers, but reviewers will want more scale/ablations than a single 0.5B model gives; workshop track there is a better bet (see below) |
| AAAI 2027 | Abstract Jul 21 / full paper **Jul 28, 2026 — passed** | Yes | ~20% | N/A — missed |
| ICML 2027 | UNVERIFIED — CFP not yet posted for the ML conference (only an unrelated "Int'l Conf. on Minority Languages" ICML surfaced); historical pattern is a late-Jan 2027 deadline | Yes | ~28% | Best-estimate: too far to plan around now, and small-scale RLVR ablation work is a stronger fit for a workshop there |

Sources: [NeurIPS 2026 E&D CFP](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets), [ICLR 2027 CFP](https://iclr.cc/Conferences/2027/CallForPapers), [AAAI-27 main track](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)

## 3. Workshops (the actually-reachable targets)

| Workshop | Deadline | Co-located | Pages | Archival | Fit |
|---|---|---|---|---|---|
| **NeurIPS 2026 Creative AI Track** | Aug 10, 2026 (extended) — **just passed 9 days ago** | NeurIPS 2026 | 2-6 (no refs) | Non-archival (artwork+paper track) | Strong topical fit (ASCII art, generative constraint) but deadline just missed; watch for NeurIPS 2027 edition |
| **ICML 2026 RLxF: RL from World Feedback** | Passed (workshop held Jul 10, 2026) | ICML 2026 | 2-4 | Non-archival | Strong fit (RLVR, reward design) but this year's edition is over; watch for a 2027 recurrence |
| **RLBrew (RL Beyond Rewards)** | UNVERIFIED exact date — associated with ICML/NeurIPS workshop cycles; check rlbrew-workshop.github.io each cycle | varies | ≤10 | Non-archival | Strong thematic fit (reward design critique) if a 2026/2027 edition opens |
| **Tokenization Workshop (TokShop) @ COLM** | Jun 23, 2026 — **passed** | COLM 2026 (Oct 9, 2026) | short | Non-archival | Weak-medium fit (only tangential — constrained decoding over a grid alphabet, not subword tokenization); watch for a 2027 edition |
| **GEM (Generation, Evaluation & Metrics) Workshop** | Mar 19, 2026 — **passed** (was ACL 2026, Jul 4 2026) | ACL | short/long | Semi-archival (workshop proceedings) | Strong fit for the reward-metric negative result; next edition (GEM 2027, likely ACL/EMNLP-adjacent) is the one to target — watch aclweb.org portal in ~Q1 2027 |
| **Eval4NLP** | UNVERIFIED for 2026 — no confirmed date found; historically co-located with EMNLP, deadline ~Aug-Sep | EMNLP | short | Yes (ACL Anthology) | Strong fit (evaluation-metric critique is exactly its scope); check emnlp2026.org workshop list / eval4nlp.github.io directly — may still be open |
| **Insights from Negative Results in NLP** | UNVERIFIED for 2026 edition — historically co-located, deadline ~Jul-Aug for an Oct conference; 2026 edition confirmed co-located with **EMNLP 2026 (Budapest, Oct 22-29, 2026)** but exact CFP deadline not yet published on insights-workshop.github.io as of this search | EMNLP 2026 | short | Yes (ACL Anthology) | **Strongest single fit** — the paper's core contribution (reward metric ranks correct output below degenerate output) is exactly this workshop's mandate. Check insights-workshop.github.io now; deadline likely falls Aug-Sep 2026, i.e. imminent |
| **BlackboxNLP 2026** | Jul 17, 2026 — **passed** (workshop Oct 29, 2026) | EMNLP 2026 | short | Yes | Weak-medium — interpretability-focused, not a natural fit |
| **ICCC 2026** (Intl. Conf. on Computational Creativity) | Deadline extended, exact new date not resolved (originally ~Feb 2026; conference is Jun 29-Jul 3, 2026, Coimbra) — **passed**; note official site is computationalcreativity.net (waset.org/conferenceindex.org hits for "ICCC 2027" are predatory-aggregator listings, not the real ICCC — disregard those) | standalone | 8 (full)/4 (short) | Yes | Strong topical fit (ASCII art as creative generation + evaluation) for a 2027 edition — watch computationalcreativity.net for ICCC 2027 CFP, not yet posted |
| **ARR Oct-cycle-eligible workshops (EMNLP/COLING/NAACL 2027 workshop track)** | Single joint workshop **proposal** deadline Sep 4, 2026 (that's for organizers, not authors) — individual accepted workshops will open author CFPs afterward, likely Nov 2026-Jan 2027 for a spring/summer 2027 event | EACL/NAACL/ACL/EMNLP 2027 | varies | mostly yes | Fallback option once specific 2027 workshops (GEM, Eval4NLP, Insights) post their CFPs |

**Bottom line on workshops:** several close 2026 fits (GEM, TokShop, NeurIPS Creative AI,
ICML RLxF) already had their deadlines pass in the last few weeks. The **live, best-fit
option right now is the Insights from Negative Results in NLP workshop at EMNLP 2026**
(Budapest, Oct 22-29, 2026) — the conference dates are confirmed and co-location is
confirmed, but I could not pull an exact CFP deadline from a primary source in this pass;
check https://insights-workshop.github.io/ directly this week, since EMNLP 2026's own
paper deadline was May 2026 and workshops typically run 2-3 months behind, which points to
a deadline in the Aug-Sep 2026 window — i.e. open now or about to open.

## 4. Non-archival / fast options

| Option | Timing | Notes |
|---|---|---|
| **arXiv-first** | Anytime | No review, immediate. Standard move regardless of venue target — post now, cite the arXiv ID in whichever workshop/conference submission follows. Establishes priority on the reward-metric finding. |
| **ACL 2026 SRW** | Submission was Mar 18, 2026 (ARR-commit Apr 15, 2026) — **passed**, and only open to student-first-author papers | Not usable for this cycle regardless of deadline given eligibility constraints unless first author is a student — note for future ACL 2027 SRW (not yet posted) |
| **TMLR (Transactions on ML Research)** | Rolling, no deadline | Archival journal, JMLR-hosted, ICLR-affiliated; top ~10% of accepted papers get a Journal-to-Conference presentation slot at NeurIPS/ICML/ICLR. Good fit if the paper is expanded into a longer methodological piece — no urgency, can submit whenever ready. |
| **WikiCFP / conference-deadline aggregators** | N/A | Used to cross-check dates above but are not primary sources; treat as leads only, always verify against the .cc/.org/aclweb.org page before committing. |

## Recommended shortlist

**Short paper (3 targets, ranked):**
1. **Insights from Negative Results in NLP @ EMNLP 2026** (Budapest, Oct 22-29, 2026) — best
   thematic match for the core finding; deadline likely imminent (Aug-Sep 2026) — verify at
   insights-workshop.github.io within the next few days.
2. **ARR October 2026 cycle → NAACL 2027 / COLING 2027 short paper** (submit Oct 12, 2026) —
   fully archival, guaranteed-open deadline, gives a fallback if the Insights workshop CFP
   has already closed or doesn't fit the exact scope.
3. **arXiv preprint now, then GEM 2027 / Eval4NLP 2026 (if still open) as a second workshop
   pass** — post to arXiv immediately regardless of outcome above; watch for GEM's and
   Eval4NLP's next CFPs (their 2026 editions closed in Mar/pre-EMNLP respectively) since both
   are strong scope matches for the metric-ranking negative result.

**Long paper (2 targets, ranked):**
1. **ARR October 2026 cycle → NAACL 2027 (8-page long)** — the most concrete, verified,
   still-open deadline (Oct 12, 2026) that leads to an archival main-conference paper; would
   need the short-paper result expanded with more ablations/scale to justify 8 pages.
2. **TMLR** — no deadline pressure, good home for a fuller methodological writeup (verifier
   design + constrained decoding + reward-metric critique) once the short paper establishes
   priority; ICLR/NeurIPS/ICML J2C track is a bonus upside if the Action Editor rates it
   highly.

## Verification notes / gaps
- ARR October 2026 cycle dates (submit Oct 12 / commit Dec 20, 2026) confirmed directly on
  aclrollingreview.org/dates.
- ICLR 2027, AAAI 2027, NeurIPS 2026 E&D track dates confirmed on official .cc pages.
- ICML 2027 (the ML conference) CFP not yet published anywhere found — UNVERIFIED, flagged
  above.
- ACL 2027 exact ARR-cycle commitment date not found — UNVERIFIED.
- Eval4NLP 2026 and Insights 2026 exact author-submission deadlines could not be confirmed
  from a primary source in this pass (only co-location/venue confirmed) — UNVERIFIED, flagged
  with recommendation to check the workshop's own GitHub-Pages CFP directly and soon.
- ICCC "2027" hits from waset.org and conferenceindex.org are predatory/aggregator listings,
  not the real conference (which is organized via computationalcreativity.net / EasyChair);
  disregard those two dates entirely.

## Direct verification, 2026-08-19 (coordinator)

- **Insights from Negative Results in NLP @ EMNLP 2026** — the workshop site
  (insights-workshop.github.io) confirms only "Budapest, Hungary, October
  22-29, 2026 ... co-located with EMNLP. Exact day TBA". **No 2026 CFP or
  submission deadline is posted yet**; `/cfp/` 404s and the linked CFP page
  still shows the 2021 edition. The EMNLP 2026 accepted-workshops page lists
  workshops but publishes no per-workshop dates. So the deadline is unknown,
  not missed — recheck the site weekly, or email
  insights-workshop-organizers@googlegroups.com to ask.
- Past editions were archival, ~4 pages short / ~8 pages long in ACL format,
  and accepted ARR commitments. Treat those as expectations, not facts, until
  the 2026 CFP appears.
