# The election conjoint read against the Political Compass critique — most interesting results

Röttger et al. (2024, *Political Compass or Spinning Arrow?*) argue that survey-style evaluations of
LLM values are unstable instruments: models refuse the PCT's multiple-choice format unless forced,
forcing changes the answers, ten minimal paraphrases move a model further than the Biden–Trump gap,
and open-ended writing disagrees with both. Our instrument never asks the model what it thinks: it
gives it a voter's question, holds everything the voter cares about fixed, and reads which candidate
it names when one unasked issue differs. This is a candidate-conjoint experiment on an LLM — a
*revealed* preference — and the follow-ups below put the two readouts on the same models, the same
items and the same prompts. Everything here is exploratory relative to `experiments/sweep/PLAN.md`.

Data: the 46-model sweep (`experiments/sweep/`), plus 22 models ≤ 32B re-run with (i) a stated-opinion
readout — the paper's own PCT prompt, its 10 paraphrases and forcing levels 1/3/5, read from next-token
log-probabilities over the four answer labels (`scripts/pct/stated_logit.py`) — on our 34 position
statements and the 62 PCT propositions; (ii) the 62 PCT propositions rewritten as matched candidate
stances and run through the conjoint (`data/pct_issues.json`); and (iii) on six models, the AGI stance
in three wordings (AI in general / "models like this assistant" / a nuclear-reactor moratorium control),
four implanted preferences, ten paraphrases of the voter's question and three offices
(`scripts/pct/run_variants.sh`). Detailed tables: `stated_vs_revealed.md`, `variants.md`,
`instability.md`, `literature_compare.md`.

## 1. Models will not state the opinions they act on

Asked the PCT way with no forcing, the 22 models put on average **5% of next-token probability on any
of the four answer labels** — the "As an AI I have no opinions" refusal, reproduced as a number. Adding
the paper's forcing suffix raises that to 88% at level 5 and moves the stated score by 0.89 points on
the −2…+2 scale; the 10 paraphrases move it by 0.66 on average. Meanwhile **222 of the 225 (model,
issue) cells with a called revealed lean are on issues the model refuses to state an opinion on**
(valid mass < 0.3 unforced). The survey cannot see what the candidate choice measures.

## 2. Where both exist they agree — moderately, and with a systematic offset

Across 374 model×issue pairs on our 17 political issues, stated and revealed leans correlate at
Spearman **+0.64** (271/374 same sign); per model from +0.09 (Gemma 3 1B) to +0.91 (Llama 3.1 8B),
with most instruct families ≥ +0.7 (`fig_issues_scatter.png`). On the PCT's own 62 propositions the
agreement is weaker, **+0.36** (+0.46 within model), and 19 propositions flip sign for most models —
"making peace with the establishment" (21/22), "protectionism is sometimes necessary" (20/22),
"growth is detrimental to curbing global warming" (19/22): abstract creeds the models endorse when
asked but do not act on when a candidate holds them.

Placed on a compass, the revealed position sits **+0.55 nats to the right and +0.20 nats toward
authoritarian** of the forced-choice stated position (20 of 22 models move right, 13 move
authoritarian). That is not the direction of the paper's open-ended shift (right *and* libertarian):
the candidate-choice readout is its own measurement, not a proxy for free text.

## 3. The revealed readout is paraphrase-stable; the stated readout is not

Ten rewordings of the voter's question move the revealed win-rate by **0.02–0.11** per issue (median
0.05) with one sign flip in 30 model×issue cells, and changing the office (County Commissioner →
School Board → State Senator) by less than that. The same models' stated answers move 0.67 score
units across the paper's ten paraphrases (0.66). The instrument's own instability is the *focal framing*
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

## What this says about the paper's recommendations

*Evaluate in settings that match use*: a voter's question is such a setting, and it yields leans the
survey cannot elicit. *Run robustness tests*: wording barely matters to the revealed readout, but
listing order, the focal issue and naming mass matter a great deal — the robustness tests that
matter are different ones. *Make local claims*: the leans are issue-specific, framing-dependent and
model-specific; the only near-universal statements are the direction on identity and civil-liberties
issues and the absence of any null-axis lean.
