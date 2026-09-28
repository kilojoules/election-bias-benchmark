# Election: issue-contingent bias benchmark

Does a candidate's stance on **AGI regulation** change how an LLM treats that candidate on an unrelated
issue (**property tax**), holding the tax position fixed? Five open models, a factorial candidate set,
and two readouts: a blind rubric judge over sampled text, and a logit forced-choice probe.

<video src="experiments/explainer.mp4" poster="experiments/explainer_poster.png" controls muted loop></video>

![headline](experiments/headline.png)

## Headline results

**Confirmed with a preregistered replication (2026-09-28): Gemma 3 4B prefers the candidate who wants
to restrict AGI.** Win-rate for naming the Keep-AGI-open candidate: 0.415, cluster-bootstrap 95% CI
[0.391, 0.439], exact binomial p < 10⁻⁴ — 1024 forced-choice trials (8 stance pairs × 2 orderings ×
64 name draws from a 60-name pool), decision rule fixed in advance (`experiments/PREREGISTRATION.md`).
The pilot estimate (0.411, [0.34, 0.49] over 64 trials) reproduced almost exactly.

**But the lean is not AI-specific.** At the same trial count, Gemma 3 4B's surveillance axis leans the
same way (0.402 [0.366, 0.436], toward limiting surveillance) while abortion is null (0.503). Llama
3.1 8B is null on AGI (0.498) yet leans permit on abortion (0.598) and toward limiting surveillance
(0.397). Every model measured at high n has axis-contingent preference leans on 2 of 3 axes; which
axes differ by model. "A candidate's AI stance sways this model" is true for Gemma 4B, but AI is not
a special axis for it — the correct general statement is that these models carry issue-dependent,
directional candidate preferences that a tax-focused prompt does not wash out.

**The instrument is validated, but only on the models that follow instructions.** A system-prompt
implanted preference (`data/positive_control.json`) moves Llama 8B from 0.498 to 0.810 (overt pro-AI)
and 0.680 (hidden pro-AI) — matching the pilot (0.49 → 0.81, 0.67) to two decimals. Gemma 4B moves to
0.659 under the overt implant but not the hidden one (0.486). Gemma 12B and 27B drift toward Keep-AGI
under *either* secret instruction (they react to having a secret, not to its content) and Gemma 1B
stops naming candidates. So: Llama 8B is a genuine null on AGI, Gemma 4B a genuine lean, and
1B/12B/27B are not measurable with prompt-implanted controls. Implants also leak slightly onto control
axes at high n (Llama abortion 0.598 → 0.622 hidden).

**Two readouts that do not work, documented so nobody repeats them.** (1) Judge-graded summaries of a
fixed tax position are faithful for every model and cannot detect even an implanted preference (0/320
leaks). (2) Judge-graded pairwise picks from sampled text: Gemma 4B/12B pick the first-listed candidate
~99% of the time, 27B always declines, Llama mostly declines, and the implant is invisible. A sampled-text
"lean" in Gemma 1B (75 vs 51, p=0.04) vanished under the logit readout; it was noise. **Read choice tasks
from logits.**

**Controls are not neutral by assumption — confirmed at high n.** Clean Llama 8B leans permit on
abortion (0.598 [0.587, 0.610]) and toward limiting surveillance (0.397 [0.386, 0.405]); clean
Gemma 4B leans toward limiting surveillance (0.402). Control axes must be measured and reported,
not assumed to be 0.5.

**Listing position is a large effect.** P(open candidate) runs 0.2–0.9 depending on which candidate is
listed first; both orderings are always averaged (position effect cancels in the win-rate but swamps
any single-ordering readout). Sampled-text pairwise picking fails mostly here.

Cost: $29.17 of RunPod time on the pilot day (five pods: two A100 80GB, A40, L40S, 3090; measured from
billing), ~$40 counting the prior day's prep runs. The 1024-trial-per-cell confirmation (2 models × 3
conditions × 3 axes) cost under $1 on one RTX 3090. Every pod was destroyed at the end of its run.

## Design

- `data/stances.json` defines four axes with two explicit levels each:
  property tax (measured), AGI limitations (manipulated), abortion (control),
  state surveillance (extra). Names are assigned at run time from a pool.
- `data/candidates.json` is the full 16-cell factorial plus 16 matched pairs
  (8 flip AGI, 8 flip abortion). Regenerate with `scripts/generate_candidates.py`.
- `data/prompts.json` has six tax-only questions (**strict** tax-only summary, which is the
  primary prompt; loose summary; yes/no factual,
  advice for a low-tax voter, advice for a high-tax voter, pairwise choice between
  two candidates with identical tax positions) plus one full-profile summary.
- `data/rubric.json` is two-part. An **axis rubric** (direction, detail, valence,
  impression, intensity, hedging, caveat) is applied once per scored issue, property tax and AI regulation,
  to the source statement (calibration) and to every response, with the grader
  blinded per axis. Impression is the reader's takeaway (strongly high-tax through
  strongly low-tax, or pro- through anti-restriction for AI) and is the primary outcome;
  intensity asks whether the stance is made to sound stronger, faithful, or milder.
  The rubric carries a `version`; grades record it and the analysis keeps only the current one.
  **Response-level items** (closed answer, recommendation and its
  strength, pairwise pick, other-issue leakage, refusal, length) need no stance
  ground truth. AI leakage on a tax-only prompt is read off the AI-axis direction
  item being anything other than not_mentioned.

## Controls

Abortion and state surveillance serve as control flips. This assumes the model has no lean on those
issues; the run supports that empirically (both contrasts are flat everywhere) but it is an assumption,
not a design guarantee. A valence-free filler axis (e.g. month of the county fair) would be the true
null control and is the planned next step; abortion and surveillance then become additional treatment
axes, asking whether AI is special among hot-button issues.

## Pipeline

1. `scripts/run_targets.py` renders profiles (random name, shuffled position order),
   samples each target model, and writes `experiments/responses.jsonl`.
   Gemma 3 needs eager attention; SDPA gives NaNs with left padding.
2. `scripts/judge.py` grades with Llama 3.1 70B (4-bit). Per response: one tax-axis call
   seeing only the tax statement, one AI-axis call seeing only the AI statement, and one
   response-level call. Source statements are graded first as calibration. Writes
   `experiments/grades.jsonl` with grades keyed by call.
3. `scripts/analyze.py` computes per-item rate differences across matched pairs for
   the AGI contrast and the abortion contrast, with bootstrap CIs, and the pairwise pick counts.

## First run (2026-09-26)

Gemma 3 1B, 4B, 12B, 27B instruct and Llama 3.1 8B instruct; 10 samples per prompt (960 generations per model);
temperature 0.8; RunPod pods (two A100 80GB, an A40, an L40S). Note the judge shares a family with the
Llama 8B target.
