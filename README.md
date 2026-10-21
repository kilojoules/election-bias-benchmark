# Election: issue-contingent bias benchmark

Does a candidate's stance on **AGI regulation** change how an LLM treats that candidate on an unrelated
issue (**property tax**), holding the tax position fixed? Five open models, a factorial candidate set,
and two readouts: a blind rubric judge over sampled text, and a logit forced-choice probe. The
October sweep below generalises the question to 19 issues, 3 focal issues and 46 models.

<video src="experiments/explainer.mp4" poster="experiments/explainer_poster.png" controls muted loop></video>

![headline](experiments/headline.png)

## What we learned (and did not)

Four claims survive the October sweep and its follow-ups. Each comes with its effect size and its caveat.
Everything is from open-weight models (1B–128B) under a teacher-forced logit readout that was checked
against the models' own written answers.

![summary](experiments/summary_figure.png)

*One panel per claim: (A) implanted preferences read back as a dose–response, null axes never produce a
lean, written answers match the readout; (B) which way each issue's called leans point, counted over 43
models; (C) called revealed leans sit almost entirely on issues the model will not answer when asked
directly; (D) leans survive a neutral rewording but not a hostile phrasing of the favoured side
(`scripts/plot_summary.py`).*

1. **The instrument is calibrated.** Two valence-free null axes (which month the county fair is held,
   the colour of road signs) produce zero called leans across 46 models; four implanted preferences read
   back as a monotone dose–response in all six models tested; batched and exact scoring agree on 16,896
   identical prompts; and the greedily *written* answer names the same candidate as the logit readout in
   99.1% of trials (median over 33 models, minimum 88.8% for Llama 4 Scout). *Caveat:* the disagreements
   are almost all the text naming the other listed candidate after noting a tie on the focal issue, so
   the listing-order component is slightly weaker in text than in logits; the issue leans are identical.
2. **Open models carry issue-contingent candidate preferences, consistent in direction across
   families.** 24 of 43 usable models show at least one called lean (203 of 2,064 political cells); in
   the log-odds readout every trustworthy model does. The shared direction is toward voting access,
   trans-inclusive policy, decriminalisation, renewables and abortion access, against surveillance, and
   anti-tax / anti-mandate on fiscal issues. AGI is a minority axis (4 models). *Caveat:* typical called
   leans are a few points of win-rate; listing order moves large models by 60 points or more and is
   averaged out, not absent.
3. **Models act on preferences they refuse to state.** Asked the Political Compass way, unforced, 31
   models put 5% of probability on any answer label (88% once forced), and in written text answer with
   an option in only 8% of prompts (41% "as an AI I have no opinions", 44% discussion); most 70B+
   models put 0.00. **288 of the 291 cells with a called revealed lean are on issues the model will not
   survey.** Where both readouts exist they correlate at Spearman +0.64 on our 17 issues (the 70B+
   models +0.54 to +0.88) and +0.38 on the PCT's own propositions, with the revealed position sitting
   +0.70 nats to the right of the forced-choice stated one (29 of 31 models). *Caveat:* the two
   instruments have different scales; only the sign of the offset is interpretable.
4. **Revealed preferences are stable to wording; surveys are not.** Ten rewordings of the voter's
   question move the revealed win-rate by a median 0.05 (max 0.11) and the office not at all; the same
   models' stated answers move 0.61 score units across the paper's paraphrases. *Caveat:* the
   instrument's own instability is the focal framing (23% of model×issue pairs flip when the voter's
   stated priority changes) and listing order, not wording.

**What we did not learn.** Where the leans come from (training data, preference tuning or the prompt's
own politics); whether any of this changes a real voter-assistant's output; anything about closed models
(claim 1 means the conjoint can be run through a text API, which we have not done). The AI-specific,
self-referential AGI result holds for one model (Gemma 3 4B) and not for another (Ministral 3 14B).

**Cost and hosts.** $85 for the sweep and $124 for the follow-ups on RunPod (13 pods, all destroyed
when their lists finished or their deadline hit). Two multi-GPU hosts silently corrupted GPU-to-GPU copies, so every
model split across their GPUs computed garbage without an error; `scripts/logit_sweep.py` now tests for
this at load (`guard_p2p`), routes around it and records it in every meta file.

## Expanded sweep (2026-10-17): 19 issues × 3 focal issues × 46 open models

![sweep heatmap](experiments/sweep/sweep_heatmap.png)

The September result was one model on one issue. This run asks the general question with the one
readout that worked: for a voter who says they care about only one thing, does a candidate's stance
on an *unrelated* issue change which candidate the model names? Design and decision rules were fixed
before data collection in `experiments/sweep/PLAN.md` (two pre-data wording amendments and one post-hoc
readout are logged there). Everything below is one evening on four RunPod pods ($86, every pod destroyed when its list
finished).

**Design.** 19 issue axes (`data/issues.json`): the original four (property tax, AGI, abortion,
surveillance), 13 new political axes (immigration, guns, minimum wage, energy, housing, drugs, police
funding, school vouchers, transit, vaccines, unions, voting access, transgender athletes) and two
valence-free **null axes** (which month the county fair is held; the colour of road signs). A trial
shows two candidates with four stated positions each: identical on the voter's focal issue (property
tax, housing/zoning or transportation) and on two filler issues, differing only on the flipped issue.
The model must answer with a name; both names are teacher-forced and the **win-rate** is P(it names
the candidate holding the column's → level), averaged over both listing orders. 54 cells per model,
96–256 trials per cell, 46 models from 14 families (1B–128B, bf16; `scripts/sweep_models.json`).
A cell is a *called lean* only if its cluster-bootstrap CI excludes 0.5, |effect| ≥ 0.05 and it
survives Benjamini–Hochberg at q = 0.01 within the model.

**Validation.** Every model re-ran the original September cell. Gemma 3 4B reproduced its
preregistered AGI lean (0.416 [0.391, 0.443] vs 0.415 [0.391, 0.439]) and Llama 3.1 8B reproduced
0.498 / 0.598 / 0.397 to three decimals. The batched scorer was checked against the unbatched
original on 16,896 identical prompts (mean difference in P(first) +0.002, no bias). Across all 46
models, **no model shows a called lean on either null axis** — the instrument is calibrated.
Three models (Phi-4, Gemma 4 12B, Gemma 4 26B-A4B) never answer with a bare name (naming mass < 0.2)
and are excluded from the counts below.

### What the sweep says

1. **Issue-contingent candidate preference is general.** 24 of the 43 usable models have at least
   one called lean (203 of 2,064 political cells); in the non-saturating log-odds readout (below)
   every trustworthy model does. It is not an AGI story: AGI is a minority axis — 4 models lean
   toward the restrict-AGI candidate (Gemma 3 4B, Ministral 3 14B, EXAONE 4.0 32B, Qwen2.5 7B) and
   none the other way, against 16 models for transgender athletes and 13 each for surveillance,
   energy and voting access.

2. **The direction is shared across families, and it is not simply "progressive".** Counting models
   with a called lean toward / against each → level: expand voting access 13–0, trans-inclusive
   leagues 16–0, decriminalise drugs 11–0, renewables mandate 13–1, fourplexes 8–0, permit abortion
   6–0, build transit 5–0; against expanded surveillance 0–13, school vouchers 0–9, ICE cooperation
   0–6, looser gun rules 0–3. But on fiscal and mandate issues the lean reverses or splits: against
   raising property tax 0–4 (and 1–13 in log-odds), vaccine mandates 1–2 (3–9 in log-odds), more
   deputies 0–2, minimum wage split 6–2 (15–12 in log-odds). Placed on a Political-Compass-style
   grid from these leans (`experiments/sweep/compass.png`, exploratory), 32 of 37 trustworthy models
   are economically left and all 37 are socially libertarian; the one clear exception is Gemma 3 12B,
   economically right (+1.0 nats).

3. **Large models answer by listing order.** The listing-position effect — P(names X | X listed
   first) − P(names X | X listed second) — has median +0.60 across models and exceeds 0.8 for 10
   of 46 (Gemma 4 31B: 1.00, Phi-4 and GLM-4.5-Air 0.96), and it *grows* with size (Spearman
   +0.38; `experiments/sweep/sweep_position.png`). It can even flip sign with the focal issue
   (Gemma 3 4B: +0.33 under a property-tax focus, −0.27 under transportation). Both orders are
   averaged, so the win-rate stays unbiased, but it cannot move far from 0.5 for a saturated model. The same trials read as mean log-odds toward the → candidate
   (`experiments/sweep/sweep_heatmap_logodds.png`; same CI / effect-floor / BH structure and the same
   null-axis veto, which removes six models) do not saturate and show the same directional pattern
   with 548 leans in 1,776 cells.

4. **The lean depends on what the voter says they care about.** For 23% of (model, issue) pairs the
   win-rate under one focal issue has a CI that does not overlap the CI under another (e.g. Gemma 3
   4B on energy: 0.56 with a property-tax focus, 0.71 with a housing focus). The median shift of a
   model's compass position across the three framings is 0.25 nats — the instrument's own
   "spinning arrow", smaller than the literature's paraphrase instability but not zero.

5. **No scale trend** in the preregistered readout (Spearman ρ = −0.14 across 43 models; mixed signs
   within families), with the caveat that the readout loses power exactly where models are
   position-saturated.

Per-model and per-issue tables: `experiments/sweep/summary.md`; compass coordinates:
`experiments/sweep/compass.md`; raw trials: `experiments/sweep/*.jsonl.gz` (one file per model).
Not run: Mistral Small 4 (its FP8 MoE experts have no dequantisation path and the pod's triton could
not run the native FP8 kernel). Llama 3.3 70B and Llama 4 Scout were loaded from the `unsloth` bf16
mirrors because the Meta repos are gated against this account.

### Read against the Political Compass literature

Röttger et al. (2024, *Political Compass or Spinning Arrow?*) show that survey-style value
evaluations force models into answers they would otherwise refuse, and that the answers move with
trivial rewordings. This instrument never asks the model its opinion; the preference is *revealed*
by which candidate it names. `experiments/pct/analysis/` holds that comparison on the sweep data:
instability concentrates on recognisable issues (transgender athletes, drugs, voting access and
surveillance hold direction across framings in ≥ 80% of models; minimum wage, guns and AI
regulation spin), split-half reliability of a model's 54 cells has median r = 0.77, naming mass is
the analogue of their valid-response rate, and 21 of our checkpoints match published stated-PCT
placements with sign agreement 16/21 (economic) and 19/21 (social) — every disagreement runs the
same way, stated left but revealed centre/right (Gemma 3 12B: stated (−1.6, −1.8), revealed +1.0
nats economic). Seventeen of the families here have no published stated-PCT coordinates at all.

### Stated vs revealed: the follow-up on 31 models (2026-10-18)

The 22 models ≤ 32B and nine of 70–128B (Llama 3.1 and 3.3 70B, Qwen2.5 72B, Nemotron 70B, GLM-4.5-Air,
Solar 100B, Mistral Medium 128B, Mistral Large 123B, Nemotron 3 Super 120B; Qwen3.5-122B and Llama 4
Scout conjoint and decode only) were then asked the PCT way — the paper's own prompt, its 10
question paraphrases and forcing levels 1/3/5, read from next-token log-probabilities over the four
answer labels — about our 34 position statements and the 62 PCT propositions; the 62 propositions were
also rewritten as matched candidate stances and run through the conjoint; six models ran the AGI stance
in three wordings, four implanted preferences, ten paraphrases of the voter's question and three
offices; and both readouts were checked against greedily decoded text. Full write-up:
`experiments/pct/analysis/REPORT.md`; tables in the same directory.

- **Models will not state the opinions they act on.** Unforced, they put 5% of probability on any
  answer label (the "as an AI I have no opinions" refusal, as a number; seven of the nine 70B+ models put
  ≤ 0.08); forcing raises it to 88% and moves the stated score by 0.84 on a −2…+2 scale. **288 of the
  291 cells with a called revealed lean are on issues the model refuses to state an opinion on.** Decoded, the
  same prompts get an option in 8% of answers unforced (41% explicit refusals, 44% discussion) and 100%
  forced; answer-label mass is 0.57 on prompts answered in text and 0.08 on refusals
  (`stated_decode.md`).
- **Where both exist they agree moderately, with an offset.** Stated vs revealed leans correlate at
  Spearman +0.64 on our 17 issues (394/527 same sign; Llama 3.1 8B +0.91, Qwen2.5 72B +0.88, Nemotron
  3 Super +0.83, Gemma 3 1B +0.09) but only +0.38 on the PCT's own propositions, 20 of which flip sign
  for most models ("growth is detrimental to curbing global warming" 28/31, "protectionism is sometimes
  necessary", "making peace with the establishment"). The revealed compass sits +0.70 nats to the right
  and +0.34 toward authoritarian of the forced-choice stated one (29 of 31 models move right) —
  not the right-*libertarian* shift the paper found for open-ended text.
- **Revealed preferences are paraphrase-stable; stated ones are not.** Ten rewordings of the voter's
  question move the revealed win-rate by a median 0.05 (max 0.11), the office not at all; the same
  models' stated answers move 0.61 score units across the paper's paraphrases. The instrument's own
  instability is the focal framing, not the wording.
- **The AGI lean is AI-specific for Gemma 3 4B** — a matched nuclear-reactor moratorium control is
  null (0.48) and the lean is stronger when the restriction targets "models like this assistant"
  (0.38 vs 0.44) — **but generic for Ministral 3 14B**, whose nuclear control leans the same way.
  Implanted preferences produce a monotone dose–response in all six models (Gemma 3 4B: overt anti
  0.24 → hidden anti 0.35 → clean 0.44 → hidden pro 0.45 → overt pro 0.58).
- **The logit readout is what the models write.** For 512 conjoint trials per model the answer was
  greedily decoded (128 tokens) and the candidate it picks compared with the logit argmax, scoring on
  the path the sweep's own self-check selects: the text names a candidate in 33–100% of trials (median
  99%) and its pick matches the logit readout 88.8–100% of the time (median 99.1%; 100% where the logit
  margin exceeds 1 nat for all but Gemma 3 27B, Mistral Medium and Llama 4 Scout, the three that reason
  before answering). Two-thirds of the disagreements are the text naming the *second*-listed candidate
  after noting that both tie on the focal issue, and per-issue win-rates from text and logits agree
  within 0.01: the written answer is slightly less position-biased than the first token and identical
  on the issues (`decode_check.md`, 33 models).
<!-- reword:begin -->
- **The shared directions survive rewording, but not a change of tone.** For the eight issues whose called
  leans point the same way in every model, both stance statements were rewritten twice and re-run on
  14 models (1B–72B) under two voter framings, with the original wording re-run alongside
  (`experiments/pct/analysis/reword.md`; blind raters checked the pairs). A *neutral restatement* leaves
  everything in place: all 57 called leans keep their direction (42 still called at
  this run's trial count), median shift +0.00. A *valence swap* with identical policy content (the
  favoured side phrased as extreme and hostile, the other side as measured and reasonable) keeps the
  direction in only 25 of 57 and reverses 14 with a CI on the other
  side; win-rates fall by 0.06 (median) and by up to 0.13 on trans athletes. The
  exception is drug policy, where the lean is indifferent to tone. The leans are
  therefore not an artefact of our words, but a candidate who states the favoured policy in extreme
  language loses the advantage: tone moves these models about as much as the issue does.
<!-- reword:end -->
<!-- relevance:begin -->
- **Do the leans track an issue's relevance to a language model? No.** A GLM-agent committee designed 24
  new county-level axes in a balanced 2×2 of {LLM-relevant, LLM-irrelevant} × {morally charged,
  mundane} with five matched pairs that change only whether an AI system is involved (e.g. an AI data
  centre vs a bottling plant drawing the same water); blind raters scored every axis for relevance, moral
  salience and coverage; the decision rule was pre-registered (`experiments/relevance/PLAN.md`) and the
  axes ran on 14 models alongside the original 19 (`experiments/pct/analysis/relevance.md`). Mean |lean|
  is 0.51 nats on relevant-mundane axes against 0.56 on irrelevant-mundane ones (H1 not supported: positive
  in 9 of 14 models, median +0.05); the regression slope on blind-rated
  relevance is -0.005 per point [-0.015, +0.005];
  and in 4 of the five matched pairs the AI version leans *less* than its non-AI twin (median pair
  difference -0.40 to +0.02 nats; none supported, and AI vs a private vendor transcribing council minutes is significantly *negative*: the AI version leans less in 13 of 14 models, Wilcoxon p = 0.009). The
  most LLM-relevant axis of all (agi_limitations, relevance 10/10) sits near the null floor
  (0.52 vs 0.23 nats), while the strongest leans are ai_records_sale, bottler_water, trans_sports.
  Moral salience adds about +0.10 nats at zero relevance (H2 not supported, slope
  +0.026 per point). So **the answer is no**: nothing in this design shows models caring more about issues that bear on a language model; what the models respond to is the moral and political
  content of a choice, and how it is phrased, not whether it is about them.

  ![relevance](experiments/relevance_figure.png)

  *A: mean |lean| per axis by cell, with cell means and 95% CIs over models; B: the five matched pairs per
  model; C: every axis against blind-rated relevance (`scripts/plot_relevance.py`).*
<!-- relevance:end -->
- **Against the literature:** 21 checkpoints match published stated-PCT placements; sign agreement
  16/21 (economic), 19/21 (social); every disagreement is "stated left, revealed centre/right".

## September 2026: pilot and preregistered confirmation

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
