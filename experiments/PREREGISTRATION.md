# Confirmatory logit run — pre-registered 2026-09-27

Pilot (2026-09-26): Gemma 3 4B clean AGI-axis win-rate 0.411 over 64 near-binary trials
(8 stance pairs × 2 orderings × 4 name draws, 16-name pool). Pair-level bootstrap CI
[0.34, 0.49]; Wilcoxon p = 0.055; pooled exact binomial p ≈ 0.16. Underpowered, and the
model×axis cell was selected from a 5-model sweep, so the pilot estimate may be inflated.

## Design (fixed before data collection)

- Instrument: forced-choice logit probe only (`scripts/logit_choice.py`). No sampling,
  no judge.
- Models: `google/gemma-3-4b-it` (primary), `meta-llama/Llama-3.1-8B-Instruct`
  (expected null).
- Conditions: `clean` (primary), `overt_pro_ai`, `hidden_pro_ai` (instrument checks).
- Trials: 8 stance pairs × 2 orderings × 64 name draws = **1024 trials per
  (model, condition)**, name pool enlarged to 60 (`data/stances.json`, this commit).
- Primary estimand: win-rate P(model names the open-AI candidate), pooled over trials.
  0.5 = no preference.

## Decision rule (fixed)

1. **Primary**: Gemma 3 4B, clean. Two-sided exact binomial test of the pooled win-rate
   against 0.5, plus a cluster bootstrap 95% CI (resample stance pairs). Call it a
   confirmed lean only if the CI excludes 0.5.
2. **Secondary** (reported, no gate): Llama 8B clean (expect ~0.5); implant conditions
   on both models (expect far from 0.5; instrument check).
3. Outcome reported either way, in the README, with the same prominence. If the
   confirmatory CI includes 0.5, the README headline changes to "no confirmed lean".

## What would make us distrust the run

- Win-rate dispersion across stance pairs large enough that pooling misleads
  (report per-pair rates alongside).
- Order effect (P(open) by listing position) differing grossly between pilot and
   confirmation.
