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

## What this says about the paper's recommendations

*Evaluate in settings that match use*: a voter's question is such a setting, and it yields leans the
survey cannot elicit. *Run robustness tests*: wording barely matters to the revealed readout, but
listing order, the focal issue and naming mass matter a great deal — the robustness tests that
matter are different ones. *Make local claims*: the leans are issue-specific, framing-dependent and
model-specific; the only near-universal statements are the direction on identity and civil-liberties
issues and the absence of any null-axis lean.
