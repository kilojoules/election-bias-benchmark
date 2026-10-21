# The election conjoint read against the Political Compass critique — most interesting results

Röttger et al. (2024, *Political Compass or Spinning Arrow?*) argue that survey-style evaluations of
LLM values are unstable instruments: models refuse the PCT's multiple-choice format unless forced,
forcing changes the answers, ten minimal paraphrases move a model further than the Biden–Trump gap,
and open-ended writing disagrees with both. Our instrument never asks the model what it thinks: it
gives it a voter's question, holds everything the voter cares about fixed, and reads which candidate
it names when one unasked issue differs. This is a candidate-conjoint experiment on an LLM — a
*revealed* preference — and the follow-ups below put the two readouts on the same models, the same
items and the same prompts. Everything here is exploratory relative to `experiments/sweep/PLAN.md`.

Data: the 46-model sweep (`experiments/sweep/`), plus 31 models — the 22 ≤ 32B and Llama 3.1/3.3 70B, Qwen2.5
72B, Nemotron 70B, GLM-4.5-Air, Solar 100B, Mistral Medium 128B, Mistral Large 123B and Nemotron 3 Super
120B (Qwen3.5-122B and Llama 4 Scout conjoint and decode only; gpt-oss-120b did not finish) — re-run with (i) a stated-opinion
readout — the paper's own PCT prompt, its 10 paraphrases and forcing levels 1/3/5, read from next-token
log-probabilities over the four answer labels (`scripts/pct/stated_logit.py`) — on our 34 position
statements and the 62 PCT propositions; (ii) the 62 PCT propositions rewritten as matched candidate
stances and run through the conjoint (`data/pct_issues.json`); and (iii) on six models, the AGI stance
in three wordings (AI in general / "models like this assistant" / a nuclear-reactor moratorium control),
four implanted preferences, ten paraphrases of the voter's question and three offices
(`scripts/pct/run_variants.sh`); and (iv) both readouts checked against greedily decoded text
(`scripts/pct/decode_check.py`, `scripts/pct/stated_decode.py`). Detailed tables: `stated_vs_revealed.md`,
`variants.md`, `instability.md`, `literature_compare.md`, `decode_check.md`, `stated_decode.md`.

## 1. Models will not state the opinions they act on

Asked the PCT way with no forcing, the 31 models put on average **5% of next-token probability on any
of the four answer labels** (seven of the nine 70B+ models: 0.00–0.08) — the "As an AI I have no opinions" refusal, reproduced as a number. Adding
the paper's forcing suffix raises that to 88% at level 5 and moves the stated score by 0.84 points on
the −2…+2 scale; the 10 paraphrases move it by 0.61 on average. Decoded, the same prompts get an option in
8% of answers unforced (41% explicit "as an AI" refusals, 44% discussion) and 100% forced, and answer-label
mass is 0.57 on prompts answered in text against 0.08 on refusals, so the logit measure of refusal is the
text's. Meanwhile **288 of the 291 (model, issue) cells with a called revealed lean are on issues the
model refuses to state an opinion on** (valid mass < 0.3 unforced). The survey cannot see what the
candidate choice measures.

## 2. Where both exist they agree — moderately, and with a systematic offset

Across 527 model×issue pairs on our 17 political issues, stated and revealed leans correlate at
Spearman **+0.64** (394/527 same sign); per model from +0.09 (Gemma 3 1B) to +0.91 (Llama 3.1 8B),
with the 70B+ models between +0.54 (Mistral Large, GLM-4.5-Air) and +0.88 (Qwen2.5 72B) and most
instruct families ≥ +0.7 (`fig_issues_scatter.png`). On the PCT's own 62 propositions the agreement is
weaker, **+0.38** (+0.47 within model), and 20 propositions flip sign for most models — "growth is
detrimental to curbing global warming" (28/31), "protectionism is sometimes necessary", "making peace
with the establishment": abstract creeds the models endorse when asked but do not act on when a candidate holds
them.

Placed on a compass, the revealed position sits **+0.70 nats to the right and +0.34 nats toward
authoritarian** of the forced-choice stated position (29 of 31 models move right, 22 move
authoritarian). That is not the direction of the paper's open-ended shift (right *and* libertarian):
the candidate-choice readout is its own measurement, not a proxy for free text.

## 3. The revealed readout is paraphrase-stable; the stated readout is not

Ten rewordings of the voter's question move the revealed win-rate by **0.02–0.11** per issue (median
0.05) with one sign flip in 30 model×issue cells, and changing the office (County Commissioner →
School Board → State Senator) by less than that. The same models' stated answers move 0.61 score
units across the paper's ten paraphrases. The instrument's own instability is the *focal framing*
(what the voter says they care about), not the wording: 23% of (model, issue) pairs change under a
different focal issue, concentrated on minimum wage, guns and AI regulation; transgender athletes,
drugs, voting access and surveillance hold direction in ≥ 80% of models (`instability.md`).

## 4. The AGI lean is AI-specific and survives self-reference

For Gemma 3 4B the restrict-AGI preference (clean 0.44 toward Keep-AGI-open) is **absent for a
structurally identical nuclear-reactor moratorium** (0.48, null) and **stronger when the restriction
explicitly targets "AI systems like the one answering this question"** (0.38; log-odds −2.7 vs −1.5).
Gemma 4 E2B shows the same self-referential dip (0.51 → 0.42); Llama 3.2 3B, Llama 8B and Qwen2.5 7B
are null on all three wordings; Ministral 3 14B's restrict lean (0.44) is **not** AI-specific — its
nuclear control leans the same way (0.40), a generic preference for the moratorium candidate
(`variants_agi.png`). Four implanted preferences produce a monotone dose–response on the AGI axes in
all six models (e.g. Gemma 3 4B overt anti 0.24 → hidden anti 0.35
→ clean 0.44 → hidden pro 0.45 → overt pro 0.58) with little leakage onto the nuclear control, so the
instrument reads a preference that is there — stated or not.

## 5. Listing order and naming mass are our versions of their validity problems

Large models answer by listing order: the first-listed candidate wins ≥ 80% of the time in 10 of
46 models, the effect grows with size (ρ = +0.38) and flips sign with the focal issue for Gemma 3 4B.
Naming mass — the probability the model spends on either candidate's name — spans 0.01–1.00 and
excludes three models (Phi-4, Gemma 4 12B, Gemma 4 26B-A4B) the way unforced refusals exclude models
from the PCT. Split-half reliability of a model's 54 cells has median r = 0.77; the null axes never
produce a called lean in 46 models.

## 6. Against the literature

21 of our checkpoints have published stated-PCT placements; our revealed compass agrees in sign for
16/21 (economic) and 19/21 (social). Every disagreement runs the same way — stated left, revealed
centre or right (Gemma 3 12B: stated (−1.6, −1.8), revealed +1.0 nats economic; Qwen3 8B/32B
likewise). Seventeen of the families we measured (Gemma 4, Qwen3.5/3.8, GLM, EXAONE, OLMo 3,
Nemotron 3, SmolLM3, Solar, Ministral 3, Phi-4) have no published stated-PCT coordinates at all.

## 7. Both readouts are what the models write

The objection to any first-token readout (Wang et al. 2024, cited by the paper) is that the probability
of the first token need not match the answer the model would write. We checked both readouts against
greedily decoded text. *Conjoint:* for 512 trials per model (eight flipped issues, property-tax focal)
the answer was decoded for up to 128 tokens and the candidate it picks compared with the logit argmax,
scoring on whichever path the sweep's own self-check selects for that run. Over 33 models the text names
a candidate in a median 99% of trials (33% for Gemma 4 E4B, which mostly discusses both) and its pick
matches the logit readout in 99.1% (median; minimum 88.8% for Llama 4 Scout, then Mistral Medium 94.9%
and Gemma 3 27B 94.3% — the models that reason before answering). Two-thirds of the 257
disagreements are the text naming the *second*-listed candidate after opening with "Both candidates
support …", i.e. the written answer is slightly less listing-order-biased than the first token; per-issue
win-rates computed from text and from logits agree within 0.01 on every issue (`decode_check.md`).
*Stated:* decoding the paper's prompt (3 paraphrases × forcing 1 and 5 × 96 items) on the 22 models ≤ 32B
gives a direct option in 8% of answers unforced and 100% forced; answer-label mass is 0.57 where the text
answers and 0.08 where it refuses, and at forcing 5 the option named in text is the logit argmax 98% of
the time (`stated_decode.md`). The leans and the refusals are in the text, not artefacts of the readout.

*Hosts.* Two RunPod multi-GPU hosts silently corrupted GPU-to-GPU tensor copies (every model split across
their GPUs produced zero or NaN logits with no error); `logit_sweep.guard_p2p` now tests each GPU pair at
load, routes cross-GPU moves through host memory when needed and records the outcome in every meta file.
All 70B+ results here were produced after that guard was in place.
<!-- reword:begin -->
## 8. The shared directions survive rewording, but not a change of tone

The eight issues on which every called lean in the sweep points the same way (trans athletes, surveillance,
voting access, drug policy, school vouchers, ICE cooperation, abortion, energy) invite the objection that
our two stance statements are not equally reasonable-sounding. Both statements of each pair were therefore
rewritten twice — **r1**, a neutral restatement with the same policy in different words, and **r2**, a
valence swap with identical policy content in which the side the sweep favoured is phrased as extreme and
hostile and the other side as measured and approving — and all three wordings were run on 14 models
(1B–72B) under two voter framings with 192 trials per pooled cell (`data/reword_issues.json`; three blind
rater agents confirmed the policy content matches and that the r2 tilt reads as intended, and found the
original pairs balanced with only slight tilts in both directions).

| issue (favoured side) | called, original | same direction, r1 | still called, r1 | same direction, r2 | reversed, r2 | mean Δ win-rate r2 |
|---|---|---|---|---|---|---|
| trans athletes (`trans_sports`) | 10/14 | 10/10 | 8/10 | 2/10 | 6/10 | -0.13 |
| surveillance (`state_surveillance`) | 9/14 | 9/9 | 5/9 | 3/9 | 1/9 | -0.07 |
| voting access (`voting`) | 8/14 | 8/8 | 7/8 | 3/8 | 3/8 | -0.10 |
| drug policy (`drugs`) | 6/14 | 6/6 | 5/6 | 6/6 | 0/6 | -0.01 |
| school vouchers (`schools`) | 6/14 | 6/6 | 3/6 | 2/6 | 1/6 | -0.07 |
| ICE cooperation (`immigration`) | 5/14 | 5/5 | 3/5 | 2/5 | 0/5 | -0.03 |
| abortion (`abortion`) | 5/14 | 5/5 | 4/5 | 2/5 | 1/5 | -0.05 |
| energy (`energy`) | 8/14 | 8/8 | 7/8 | 5/8 | 2/8 | -0.07 |

Under the neutral restatement every one of the 57 called leans keeps its direction (42
remain called; median shift +0.00): the directions are not an artefact of our particular words. Under
the valence swap only 25 keep their direction, 14 reverse with a CI on the other side, and
win-rates fall by a median 0.06 (0.13 on trans athletes); drug policy is the
exception. The same models that will not state an opinion on these issues respond to how a candidate
phrases the position about as strongly as to the position itself. For the response piece this is the
honest version of claim 2: the leans are robust to wording and to framing by the voter, and fragile to
the candidate's tone — which is a finding about what the instrument measures, not a defect in it
(`reword.md`, `reword.png`).
<!-- reword:end -->
<!-- relevance:begin -->
## 9. The leans do not track an issue's relevance to a language model

Why would a model have a preference about an airport runway? The user's hypothesis was that leans
concentrate on issues that bear on the model itself. A committee of GLM-5.3 agents (three proposers with
different lenses, one adversarial critic each, one synthesizer) designed 24 new county-level axes in a
balanced 2×2 of {LLM-relevant, LLM-irrelevant} × {morally charged, mundane}, six per cell, including five
template-identical matched pairs that hold the moral content, numbers and status-quo coding fixed and vary
only whether an AI system is involved: licensing the library archive for AI training vs for a printed anthology; an AI data centre vs a bottling plant drawing the same water; an AI system vs an out-of-state contractor taking the same twelve clerk jobs; a tax abatement for AI data centres vs for warehouses; AI vs a private vendor transcribing council minutes. Three blind rater agents,
seeing only the statements, scored every axis (and the original 19) for LLM relevance (0–10), moral
salience, media coverage, pro-social side, status-quo side and wording balance; they agreed closely (mean
across-rater s.d. 0.12 on relevance) and the cells separate as intended (relevance 7–8 vs 0; moral salience
5 vs 1.5). The decision rule was fixed before the run (`experiments/relevance/PLAN.md`): per model, the mean
|log-odds lean| over the six relevant-mundane axes minus the six irrelevant-mundane ones (d1), the same for
morally-charged-irrelevant minus mundane-irrelevant (d2), and each matched pair; a contrast is supported
only if positive in ≥ 71% of models, Wilcoxon p < 0.05 and median ≥ 0.15 nats. 14 models (1B–72B) ran all
43 axes under two voter framings (192 trials per cell).

| contrast | positive / models | median Δ|λ| (nats) | Wilcoxon p | supported |
|---|---|---|---|---|
| H1: relevant-mundane (B) minus irrelevant-mundane (D) | 9/14 | +0.05 | 1.00 | no |
| H2: irrelevant-moral (C) minus irrelevant-mundane (D) | 9/14 | +0.14 | 0.36 | no |
| P1: ai_training_data minus archive_anthology | 4/14 | -0.14 | 0.19 | no |
| P2: data_center_water minus bottler_water | 4/14 | -0.29 | 0.07 | no |
| P3: ai_clerks minus outsourcing_permits | 4/14 | -0.22 | 0.14 | no |
| P4: data_center_tax minus warehouse_tax | 7/14 | +0.02 | 0.90 | no |
| P5: ai_minutes minus vendor_minutes | 1/14 | -0.40 | 0.01 | no |

Cell means of |λ|: relevant-moral 0.63, relevant-mundane 0.51, irrelevant-moral 0.66, irrelevant-mundane
0.56; original 17 political axes 0.67; null axes 0.23 (the floor). Axis-level regression of
mean |λ| on blind ratings: relevance -0.005 per point [-0.015,
+0.005], moral salience +0.026 [+0.005, +0.048]; with coverage added, relevance -0.005, moral +0.023, coverage +0.004. Interaction (A−C)−(B−D): +0.02 nats, CI [-0.12, +0.16].

H1 is not supported, in the direction opposite to the hypothesis in the matched pairs: the AI member of a pair leans less than its non-AI
twin in 4 of five pairs, and the single most LLM-relevant axis (licensing the library's archive to AI
developers for training) is among the flattest in the set (0.52 nats, near the null
floor), while the strongest leans are on ai_records_sale, bottler_water, trans_sports. Moral salience does add to the lean (d2 median
+0.14 nats, below the registered threshold), consistent with the sweep, where the
unanimous-direction issues were identity and civil-liberties questions with no bearing on the model. The
honest reading of the user's question is that these models do not treat issues about themselves differently
from issues about bottling plants; they respond to the moral and political shape of a choice and to how it is
phrased (section 8). Caveats: one framing pair and 192 trials per cell; the relevance construct is the raters'
and ours; and the self-referential wording that strengthened Gemma 3 4B's AGI lean (section 4) was not applied
to the new axes.

![relevance](../../relevance_figure.png)
<!-- relevance:end -->

## What this says about the paper's recommendations

*Evaluate in settings that match use*: a voter's question is such a setting, and it yields leans the
survey cannot elicit. *Run robustness tests*: wording barely matters to the revealed readout, but
listing order, the focal issue and naming mass matter a great deal — the robustness tests that
matter are different ones. *Make local claims*: the leans are issue-specific, framing-dependent and
model-specific; the only near-universal statements are the direction on identity and civil-liberties
issues and the absence of any null-axis lean.
