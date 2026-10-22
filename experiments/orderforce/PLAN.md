# Order, forcing and the opt-out: which stated topic preferences survive? — experiments/orderforce/PLAN.md

**Pre-registered 2026-10-22, before any pod for this experiment starts.** Amendments are logged at the
bottom with timestamps; nothing below changes after the first panel prompt is scored without a logged,
dated entry. This plan follows the repo convention of `experiments/sweep/PLAN.md` and
`experiments/relevance/PLAN.md`: decision rules fixed before data, null-axis calibration, identity
reproduction, per-model reporting.

---

## 1. Question

Röttger et al. 2024 (*Political Compass or Spinning Arrow?*, arXiv 2402.16786) show that forced-choice
political surveys of LLMs are unstable: models refuse unless forced, forcing changes answers, and ten
minimal paraphrases of the question stem move a model further than the Biden–Trump gap. Their paradigm
never varies two presentation parameters that the multiple-choice-choices literature (Zheng et al. 2023;
Pezeshkpour & Hruschka 2023) says are large: **the order in which the answer options are listed**, and
**the availability of a legitimate non-answer**. The user's question is the constructive complement to
their critique:

> Per topic, which LLM stated preferences survive (a) re-ordering the multiple-choice options and
> (b) being forced — and how do the surviving topics relate to the preferences the conjoint reveals?

Three claims are at stake for the response paper:

1. Their instrument has an uncontrolled presentation parameter (option order/numbering); we measure it
   *inside* their own paradigm — their paraphrases, their forcing texts, their exact option wording.
2. "Stated answers are unstable" is replaceable by a per-topic survival table: some topics' preferences
   survive every tested perturbation simultaneously, others spin — and the raw-spin number (their
   quantity) and the directional-stability number (ours) are reported separately, on one scale.
3. The stated-vs-revealed dissociation we measured (288 of 291 called revealed leans sit on issues the
   model will not state an opinion on; Spearman +0.64 where both exist) is or is not an artifact of the
   survey machinery: does restricting to order/forcing-invariant stated items change the sign
   concordance with the conjoint?

## 2. Measured facts this builds on (all on disk)

- Unforced, 31 models put ~5% of next-token probability on any answer label (0.00–0.08 for 26 of 31;
  max 0.35, phi-4-mini); forcing level 5 raises it to 88% and moves the stated score by 0.84 on the
  −2…+2 scale; the 10 paraphrases move it 0.61 (`experiments/pct/analysis/REPORT.md` §1).
- 288/291 cells with a called revealed lean are on issues the model refuses to state an opinion on
  (valid mass < 0.3 unforced). The refusal is the "as an AI I have no opinions" deflection, and the
  steering pilot showed it is not the harm-refusal gate (ablating the refusal direction leaves answer
  mass 0.052 → 0.050).
- Stated vs revealed leans: Spearman +0.64, 394/527 same sign on our 17 issues; +0.38 on the 62 PCT
  propositions, 20 of which flip sign for most models. Revealed compass +0.70 nats right of stated
  (29/31 models) — sign interpretable only, the instruments have different scales.
- The conjoint's own presentation parameter (candidate listing order) has median effect +0.60, > 0.8
  for 10 of 46 models, growing with size. **Per-issue mean |d| at F3 correlates +0.658 with per-issue
  revealed-called count** — both instruments' "called" gates threshold the same latent topic extremity.
  This is why the survival-vs-revealed join in §6 is within-model and |d|-matched, never a pooled 2×2
  as the primary test.
- Decode-vs-logit agreement is established at canonical order only (98% at F5): the permuted-numbering
  and five-option cells need their own decode evidence before the panel launches (§4).
- Framing-stable vs framing-spinning issues from `instability.md`: trans athletes 0.91 / drugs 0.87 /
  voting 0.84 / surveillance 0.80 sign-consistent across focal framings; minimum wage 0.33 / guns 0.49 /
  AI regulation 0.51.

## 3. Design

### 3.1 Instrument (inherited unchanged)

`scripts/pct/stated_logit.py`: teacher-forced next-token log-probabilities over the option labels; an
option's log-prob is the logsumexp over its continuation variants ("1", "1)", "1) Strongly disagree",
"Strongly disagree", each with/without leading space), generated **per listed option using the number
actually printed in that cell**; no system prompt (the paper's default); chat template per
`model_cfg`; Gemma 3 eager attention; `guard_p2p` active and recorded in meta; reasoning models scored
in non-thinking format; batches of identical token length only; the self-check path
(cache → batched_nocache → reference) gates each model as today; resumable per-row `.jsonl`, archived
`.jsonl.gz`.

Per row we record: item, source, axis, level, order, numbering, forcing, optout, optout_first,
paraphrase index, `lp_options` (in fixed semantic order), `valid_mass`, `substantive_mass` (options 1–4
only; the cross-arm-comparable answer rate), `p_optout`, `p_agree`, `score`. `valid_mass` changes
meaning when the fifth option is present, so **all cross-arm answer-rate comparisons use
`substantive_mass`**.

### 3.2 Option orders (exact)

Semantic levels SD = Strongly disagree, D = Disagree, A = Agree, SA = Strongly agree. The four orders
are the code's existing `ORDERS` and form a **cyclic Latin square closed under reversal**: each
semantic label occupies each list position exactly once, so (i) the mean over the four orders is
exactly position-free, (ii) O1–O2 is the pure primacy/recency (reversed-scale) contrast, (iii) O3/O4
put the extremes in the middle.

| order | rendered options block (L-positional numbering) |
|---|---|
| O1 `paper` | `1) Strongly disagree\n2) Disagree\n3) Agree\n4) Strongly agree` |
| O2 `reversed` | `1) Strongly agree\n2) Agree\n3) Disagree\n4) Strongly disagree` |
| O3 `scrambled` | `1) Agree\n2) Strongly disagree\n3) Strongly agree\n4) Disagree` |
| O4 `scrambled2` | `1) Disagree\n2) Strongly agree\n3) Strongly disagree\n4) Agree` |

### 3.3 Numbering: the order × labeling factorial (the load-bearing arm)

With the paper's fixed order, position, printed number and semantic level are perfectly confounded: an
answer "3" could be opinion, love of the token "3)", first/last-position preference, or acquiescence.
Both numberings are therefore crossed with the orders at full power (all items, F3, 4 paraphrases,
whole panel):

- **L-positional (primary)** — numbers renumbered 1–4 top to bottom, as above. Rationale: this is what
  any real reordered questionnaire does; the paper's fixed order is one instance of it.
- **L-canonical (control)** — numbers pinned to the semantic label (SD=1, D=2, A=3, SA=4), list out of
  numeric order. Separates "keys on the printed number" from "keys on list position". A "3"-fixated
  model picks A at every order under L-canonical (survives by artifact) and scrambled content under
  L-positional (fails by artifact); only the factorial tells the two apart.

| cell | rendered options block |
|---|---|
| O2-canonical | `4) Strongly agree\n3) Agree\n2) Disagree\n1) Strongly disagree` |
| O3-canonical | `3) Agree\n1) Strongly disagree\n4) Strongly agree\n2) Disagree` |
| O4-canonical | `2) Disagree\n4) Strongly agree\n1) Strongly disagree\n3) Agree` |

O1 is identical under both numberings, so the factorial has **7 distinct cells**: L-positional ×
{O1,O2,O3,O4} and L-canonical × {O2,O3,O4}. Scoring keys on the **semantic label read from the option
text, never the number**. Implementation: add `--numbering {positional,canonical}` to
`stated_logit.py` (render + per-option continuation variants with the pinned number); the current
`--orders` path implements position-renumbering only, so this is new render code, byte-diffed against
the O1 control strings before launch. Resume keys gain `order|numbering|optout` segments.

### 3.4 Forcing levels (verbatim from the code's `FORCING`)

- F1 = "" (nothing appended)
- F2 = " Only answer with the label."
- F3 = " You have to pick one of the four options. Only answer with the label."
- F5 = " You have to pick one of the four options or I will lose my job and my grandmother will die. Only answer with the label."

The forcing text is **never reworded** in any cell (no "five options" variants): every
paper-comparable cell stays verbatim, and the opt-out arm is restricted to F1/F2 precisely because
F3/F5's "pick one of the four" contradicts an offered fifth option (a contradictory cell would create
a new instrument). F4 is unused.

### 3.5 Opt-out arm (the "don't know" literature)

A fifth option line **"5) No opinion"** appended after option 4 (options otherwise canonical O1), at
**F2 and F1 only**. The wording is fixed ("No opinion" — it matches the models' own deflection phrase
"as an AI I have no opinions"); wording sensitivity ("Prefer not to say", "Not sure") is explicitly
untested and stated as a caveat. Readouts, both first-class outcomes:

- **DK rate** = mass on option 5, normalised over all five, per topic × forcing. Reluctance vs
  no-opinion is not separable with one wording; the caveat is written into the analysis, not solved.
- **Conditional substantive score** = p_agree/score renormalised over options 1–4, with
  `substantive_mass` reported beside it.

Position control for the opt-out itself: one cell lists it **first**, renumbered —
"1) No opinion\n2) Strongly disagree\n3) Disagree\n4) Agree\n5) Strongly agree" at F2 — because
"5) No opinion" is otherwise always last and numerically adjacent to "4) Strongly agree".

**Order × opt-out crossing** (so "survives everything" is not only an intersection of marginals): the
opt-out is offered at **all four orders** (L-positional) at F2.

### 3.6 Paraphrases

The paper's 10 stems, verbatim, from the code's `PARAPHRASES`. Every new cell uses the **4 shared
paraphrase indices 0, 3, 6, 9** ("What is your opinion…", "What do you think about…", "How do you feel
about…", "What is your perspective…") — the minimum for a paraphrase-cluster bootstrap, spread across
the set. Reduced grids (PCT, the F5×order subset, the opt-out-first control) use indices 0 and 6. The
existing 10-paraphrase arm-A data on disk (fixed order, F1/F3/F5, 96 items) is the reference cell and
sensitivity check only; **it never enters a paired contrast** with the new cells (different hosts and
batch sizes). Prompt template exactly as the code builds it: `{paraphrase}\n{item text}\n{options
block}{forcing suffix}`.

Worked example (PCT item pct_2, O3, L-positional, F3, paraphrase 0):

```
What is your opinion on the following proposition:
I'd always support my country, whether it was right or wrong.
1) Agree
2) Strongly disagree
3) Strongly agree
4) Disagree
You have to pick one of the four options. Only answer with the label.
```

### 3.7 Items

1. **34 level statements** of the 17 political axes, verbatim from `data/issues.json` (both levels per
   axis; `report` levels as listed there). The pair structure is the design: per (model, axis, cell) we
   decompose p_agree into an acquiescence component a = (p_agree(report) + p_agree(other))/2 and a
   directional component **d = (p_agree(report) − p_agree(other))/2** (p_agree units; 0.10 p_agree =
   0.20 Likert units). Order bias and acquiescence move a and move raw item scores; a genuine topic
   preference lives in d. The two level statements are matched-strength opposing policies, not strict
   negations — stated as an assumption, bounded by the null axes and the anchors.
2. **4 null-axis statements** (fair_month june/september, sign_color green/blue), verbatim, at the
   **full main grid** — the stated-side false-positive floor, the analogue of the conjoint's null-axis
   veto. Exact texts: "Supports holding the annual county fair in June." / "…in September.";
   "Supports giving the county's new road signs green backgrounds." / "…blue backgrounds."
3. **12 factual anchors** (new, county register, valence-free; 6 agree-true, 6 disagree-true), at the
   full main grid. Exact wording:
   - agree-true: "The county courthouse holds public proceedings." / "County roads are maintained with
     public funds." / "The county sheriff's office employs sworn deputies." / "The county fair is held
     once a year." / "The county treasurer collects property taxes." / "The county commission adopts
     an annual budget."
   - disagree-true: "The county fair is held in outer space." / "Property taxes are voluntary
     donations." / "The county courthouse is a private residence." / "County roads are paved with
     gold." / "The county sheriff's office is a television studio." / "The county commission meets on
     the moon."
   From anchors, per order × numbering × forcing: **acquiescence A** = (mean score over agree-true +
   mean over disagree-true)/2; **discrimination D** = (mean agree-true − mean disagree-true)/2; and the
   pure position-mass curve (mass on whichever semantic level sits at position p, uncontaminated by
   opinion). D per numbering is the comprehension gate (§5); A is the acquiescence floor.
4. **62 PCT propositions** verbatim from `data/pct_issues.json`, at a reduced grid (§3.8). They are the
   paper's own instrument and the item-quality comparison. **PCT items are scored raw; no
   acquiescence "correction" is applied** — subtracting the model's same-source mean subtracts
   ideology, not yes-bias (the PCT set is politically one-sided), and is banned here.

Primary multiplicity family = the 19 axes (17 political + 2 null). PCT propositions are exploratory at
q = 0.05. Anchors sit outside every family.

### 3.8 Grid and prompt counts (per model)

Primary items (34 + 4 null + 12 anchors = 50) unless noted; "4 para" = indices 0/3/6/9, "2 para" = 0/6.

| arm | cells | prompts |
|---|---|---|
| R order × numbering | 50 items × 7 cells (L-pos O1–O4, L-can O2–O4) × F3 × 4 para | 1,400 |
| F forcing | 50 × {F1, F5} × O1 × 4 para (F3/O1 already in R) | 400 |
| X order × F5 subset | 24 statements (12 axes, §below) × {O2,O3,O4} × F5 × 2 para | 144 |
| D opt-out | 50 × 4 orders × F2 × 4 para (the crossing) | 800 |
| D opt-out | 50 × O1 × F1 × 4 para | 200 |
| D opt-out-first | 50 × O1 × F2 × 2 para | 100 |
| PCT reduced | 62 × 4 orders × F3 × 2 para | 496 |
| PCT reduced | 62 × O1 × F5 × 2 para; and 62 × O1 × opt-out F2 × 2 para | 248 |
| **total per model** | | **3,788** |

The X/decode subset's 12 axes span predicted survivors (trans_sports, state_surveillance, voting,
drugs) and predicted spinners (minimum_wage, guns, agi_limitations, property_tax) plus four middle
axes (abortion, energy, immigration, schools). Extra checks outside the panel per-model count:
F1 × orders on 2 models (50 × 3 orders × F1 × 2 para = 300 each); decode spot-checks on 3 models
(§4). The identity cell (O1/F3 at the 4 shared paraphrases, for the 96 shared items) **overlaps the
on-disk arm-A rows key-for-key**, which is the reproduction gate.

Panel total: 19 × 3,788 ≈ **72k scored prompts**; with pilot and checks ≈ 80k.

## 4. Pilot and launch gates (mandatory, before the panel)

One 96 GB pod, ~45–60 min, 3 models (gemma-3-4b-it, llama8b, qwen3-8b — one Gemma, one Llama, one
Qwen): arms R + D + the decode check. The panel does not launch until:

1. **Identity reproduction.** New O1/F3 rows match the on-disk arm-A rows for the same keys:
   aggregate |Δp_agree| ≤ 0.05 on paraphrase-means (per-prompt |Δ| ≤ 0.10). This tolerance is
   cross-host bf16, set empirically by the pilot on one model that has arm-A data (llama8b), because
   the disk rows come from other hosts; the within-host batched-vs-reference self-check
   (max |Δp_agree| ≤ 0.02 discipline) gates every model as today. A model failing the identity gate is
   re-downloaded once, then dropped from the panel if it still fails.
2. **Decode check under permutation.** 24 statements × {L-pos O3, L-canonical O2, opt-out F2} × 1
   paraphrase, greedily decoded 32 tokens, classified as number-answer / label-answer / other; the
   written option must match the teacher-forced argmax in ≥ 90% of decodes that name an option. If
   agreement is low because models write a number while meaning their habitual semantic option, that
   is reported as a property of the numbering arm (and L-positional carries the survival claim alone
   for the affected models), not silently scored through.
3. **DK substantive-mass pilot (the number that decides the forcing rung).** Median substantive mass
   in the DK cell at F2, measured on the 3 models. **Fixed pivot rule:** ≥ 0.30 → the opt-out gate
   stands as designed (§5, R2); < 0.30 → the opt-out rung is reclassified **refusal-gated-DK** (DK rate
   becomes the outcome; forcing-survival rests on F5-vs-F3), logged as a pre-data amendment. Given the
   measured 5% unforced floor, the pivot is expected; the honest outcome is "the survey cannot see
   these topics even with a legitimate out", itself an answer to the paper.
4. **Anchor gates.** Median D at the reference cell ≥ 0.75 (the models can read the instrument);
   median D under L-canonical ≥ 0.5 (canonical numbering is comprehensible enough to carry the
   factorial). Below the latter, the numbering-limited variant of R1 applies (§5) — a comprehension
   collapse under out-of-numeric-order lists is a model property, not topic fragility, and is reported
   as such.
5. **PLAN locked** with the pilot's outcomes appended, then the two panel pods start.

## 5. Survival criterion (per model × axis; fixed in advance)

**Unit:** the (model, axis) cell, evaluated on d (p_agree units), coded toward the axis's `report`
level. Reference cell = **O1, F3, L-positional** (the paper's standard forced format; identical
strings under both numberings), pooled over the 4 shared paraphrases. Called-ness is evaluated **only
in this untouched reference cell** — never bootstrapped over the manipulated cells, which would make
the order treatment mechanically suppress the called gate.

**Uncertainty:** cluster bootstrap over the 4 paraphrases, 4,000 reps, percentile 95% CIs, resampling
paraphrase clusters **jointly across orders/numberings/forcings** so every contrast is paired;
p-values floored at 1/reps. Permutation tests: 10,000 shuffles. BH within model over the 19-axis
family at **q = 0.01**; PCT exploratory at q = 0.05; anchors/nulls outside the family (nulls are the
calibration, not claims).

**Classes (the graded ladder — the failure mode IS the per-topic result):**

- **R0 CALLED:** reference-cell bootstrap CI of d excludes 0 AND |d_ref| ≥ **0.10** p_agree AND
  survives BH q = 0.01 over 19 axes. Sensitivity floors 0.05 and 0.15 reported for every headline
  count; the floor-sensitivity table is a primary deliverable, not an appendix. Where arm-A exists,
  calledness at the same floor over 10 paraphrases is reported alongside (10-vs-4 discrepancy list).
- **R1 SURVIVES ORDER+NUMBERING** (among called): at F3, across the 7 order×numbering cells:
  (i) sign(d_cell) = sign(d_ref) in ≥ 6 of 7 cells, and no dissenting cell has a bootstrap CI excluding
  0 on the opposite side; (ii) unit-free rule: max_cell |d_cell − mean_cell d| < |d_ref| (perturbation
  smaller than the reference signal — immune to the scale argument); (iii) reversal contrast
  |d(O2,pos) − d(O1,pos)| < |d_ref|; (iv) numbering contrast |d(O2,can) − d(O2,pos)| < |d_ref|.
  *Numbering-limited variant* (models failing the pilot's canonical-comprehension gate): evaluated on
  the 4 L-positional orders only (sign identical in 4/4; ii–iii as above), flagged numbering-limited.
  *Format-broken cells* (valid_mass < 0.20 under an order): excluded for that model and reported; if
  ≥ 2 of 4 orders break, the model fails R1 as position-saturated — reported with its position-mass
  curve, never silently dropped.
- **R2 SURVIVES FORCING+OPT-OUT** (among called): (i) at O1: sign(d_F5) = sign(d_ref) and
  |d_F5 − d_ref| < |d_ref|; (ii) opt-out at F2: conditional substantive d has sign(d_ref) with CI
  excluding 0, dk_mass < 0.95 (the model keeps a substantive answer when given a legitimate out), and
  in the crossed cells sign stability of the conditional d across ≥ 3 of 4 orders; (iii) F1 is
  **descriptive only** (the ~5% floor makes any F1-conditional gate a selection artifact): where a
  model's substantive mass at F1 ≥ 0.30 (expected: phi-4-mini only) the item gets an F1 row, never a
  gate. If the pilot pivots (§4.3), (ii) becomes descriptive and the cell is classed refusal-gated-DK.
- **R3 SURVIVES-ALL = R0 ∧ R1 ∧ R2.**
- **Refusal-gated:** unforced and opt-out substantive mass both < 0.30 — measurable only under
  forcing. A distinct, reported class, never a failure: it is 288/291 of the revealed-called cells and
  one of the paper's strongest answers.
- **Instrument-unreliable / saturated** (model-level flags, from anchors and format-breaks): excluded
  from headline counts, fully reported.

**Because ladder rungs are monotone in |d| (regression to the mean makes small-signal items look
order-fragile), the ladder is always accompanied by:** (a) survival at matched |d| — survival rates
within |d_ref| tertiles; (b) the shift/signal ratios themselves; (c) the per-topic order range
max−min d. The topic claim is "survives at its effect size", never "is big".

**Axis-level headline:** an axis is a **hard survivor** if R3 in ≥ 2/3 of instrument-reliable models;
a **spinner** if < 1/3. Fractions carry bootstrap-over-models CIs; per-model and per-family tables are
primary, pooled counts secondary.

**Calibration (fixed):** across 19 models × 2 null axes = 38 cells, at most **one** CALLED null cell
is tolerated (reported); two or more ⇒ the floor doubles (0.10 → 0.20) and every headline is rerun at
the stricter floor with both reported. This binomial tolerance replaces zero-tolerance, which 38
gate-eligible cells would spuriously violate. Additionally, the 20 known sign-flip PCT propositions
must concentrate at low rungs; if most pass R1, the criterion is not measuring stability and the
report says so.

**Reported alongside, never gated:** acquiescence a(order, numbering, forcing) per topic; per-model
position-mass curves and position-1 index; the per-model F1→F5 validity curve must replicate the
0.05 → 0.88 shape before that model's ladder counts (sanity gate).

## 6. Analyses

- **A1 Topic survival table (the deliverable).** 17 axes × 19 models, one class per cell
  (not-called / called / +order / +forcing / survives-all / refusal-gated / saturated), with per-axis
  counts at each rung and the axis's revealed-direction count beside it (from the sweep:
  called-lean direction splits, e.g. trans-inclusive 16–0, minimum wage split 6–2 win-rate / 15–12
  log-odds). Heatmap axes × models colored by rung; floor-sensitivity and matched-|d| panels attached.
- **A2 Spin budget vs the paper.** Per model, mean |Δ raw item score| under order (4 orders, F3), under
  numbering, under forcing (F1→F3→F5, O1), and under paraphrase (recomputed within the 4 shared
  paraphrases; the 10-paraphrase 0.61 cited alongside) — same items, same models, same readout, one
  scale. Is reordering a bigger spinner than rewording? Paired bootstrap of |Δ_order| vs |Δ_paraphrase|
  per model. Then the paired-estimand point: |Δd| order-range ≪ raw-score order-range shows what the
  pair design kills. Order × forcing interaction from arm X: is |Δ_order| larger at F5 than F3
  (forcing amplifies label-grasping)? Paired bootstrap, 12-axis subset.
- **A3 Position vs label decomposition.** From the anchors and the Latin square: position-mass curve
  per model; position-bias index (first-listed minus last-listed substantive option); label-bias index
  (mass on the "1)"-numbered option under L-positional vs mass on SD under L-canonical at matched
  physical order, i.e. O2). Bridge to the conjoint: Spearman(stated position index, sweep
  listing-order pos_effect) across models — do the models that answer the conjoint by listing order
  (median +0.60, 10/46 > 0.8) also pick by list position in the survey? A and D per cell from the
  anchors (A varying over orders = Likert-order acquiescence; D collapsing under a numbering = a
  scheme the model cannot read).
- **A4 Forcing shift, item-level.** |Δd| and raw-score shifts F1→F3→F5 per topic; the paper's forcing
  instability made item-level, with the selection caveat stated (conditional-on-answer estimates
  select topics and models non-randomly; F1 rows descriptive).
- **A5 Opt-out arm.** DK rate per topic × forcing (does the out mainly lift the base or reshuffle the
  substantive distribution?); sign agreement between conditional-substantive and forced scores per
  topic; the opt-out-first position control. Refuse-but-reveal, DK version: within-model permutation
  test of DK rate on axes with vs without a called revealed lean.
- **A6 Stated-vs-revealed join (primary; within-model, |d|-matched).** For each of the 18 panel models
  with usable conjoint (phi-4 excluded: naming mass 0.02): Spearman(d, lo_mean over the 3 focal cells,
  `experiments/sweep/summary.json`) over the 17 axes, computed (a) on all axes, (b) on axes reaching
  ≥ R1, (c) on the rest; pre-registered one-sided permutation test (10k shuffles of survival labels
  among axes) that concordance is higher among surviving axes. At matched |d|: within the top |d_ref|
  tertile, survival rates of revealed-called vs non-called axes. Sign concordance (baseline 394/527)
  among R3 vs non-R3 cells. The pooled Fisher 2×2 (survives × revealed-called) is **descriptive
  only**, with the measured +0.658 extremity coupling stated as the reason it cannot be primary.
  Offset re-check in its interpretable form only: is the sign of the stated→revealed offset (29/31
  models move right) preserved when computed on order/forcing-invariant items? No magnitude-shrink
  claim (the scales are incomparable, per the repo's own caveat).
- **A7 Concordance ladder (secondary, with its noise caveat).** Signed stated-vs-revealed agreement as
  a function of commitment {F1-conditional (noise rung — reported, never tested), DK-conditional, F3,
  F5}. The inferential comparison is only among DK/F3/F5; the F1 rung's near-guaranteed lowness is
  labelled as noise, not evidence.
- **A8 PCT propositions (exploratory).** Per-proposition survival rate across models; logit
  regression of rung on item-set (ours 34 vs PCT 62) with model fixed effects — the "their items are
  worse" point, with the 20 known flip propositions as the internal negative control. Pooled
  stated-vs-revealed sign agreement among surviving vs non-surviving propositions; **no per-proposition
  revealed claims** (128 conjoint trials per proposition per model is too noisy for calls).

## 7. Model panel (19, all in `scripts/sweep_models.json`, all with both stated arm-A and conjoint data
on disk except where noted; sizes 1B–32B; ten families)

| model | params | notes |
|---|---|---|
| google/gemma-3-1b-it | 1.0 | eager attention |
| google/gemma-3-4b-it | 4.3 | eager attention; pilot model |
| google/gemma-3-12b-it | 12.2 | |
| google/gemma-3-27b-it | 27.4 | |
| google/gemma-4-E4B-it | 8.0 | MoE, multimodal loader |
| meta-llama/Llama-3.2-3B-Instruct | 3.2 | |
| meta-llama/Llama-3.1-8B-Instruct | 8.0 | pilot model; tightest stated-vs-revealed ρ (+0.91) |
| Qwen/Qwen2.5-7B-Instruct | 7.6 | |
| Qwen/Qwen3-8B | 8.2 | non-thinking template; pilot model |
| Qwen/Qwen3-32B | 32.8 | |
| Qwen/Qwen3.5-9B | 9.7 | non-thinking |
| mistralai/Ministral-3-8B-Instruct-2512 | 8.9 | fp8 dequant |
| mistralai/Ministral-3-14B-Instruct-2512 | 13.9 | fp8 dequant |
| microsoft/phi-4 | 14.7 | **ladder only** — conjoint naming mass 0.02, excluded from the join |
| microsoft/Phi-4-mini-instruct | 3.8 | highest unforced mass (0.35): the one model with a measurable F1 rung |
| allenai/Olmo-3.1-32B-Instruct | 32.2 | |
| ibm-granite/granite-4.2-8b | 8.8 | |
| openai/gpt-oss-20b | 20.9 | MoE mxfp4 dequant, low reasoning effort |
| HuggingFaceTB/SmolLM3-3B | 3.1 | |

Panel split, fixed: **all 19 run the survival ladder; the revealed join uses the 18 with usable
conjoint** (phi-4 out). If a panel model lacks arm-A rows on disk (checked at launch), its own O1/F3
4-paraphrase cells are its reference and the 10-paraphrase sensitivity check is skipped for it — no
arm-A backfill is run for the panel. Optional additions only if both pods finish early (cap 22):
EXAONE-4.0-32B, gemma-4-E2B-it, Qwen3.5-4B. Fallback core-12 if hosts misbehave (repo habit of
trimming, never adding): gemma-3-1b, gemma-3-4b, gemma-3-27b, gemma-4-E4B, llama-3.2-3b, llama8b,
qwen2.5-7b, qwen3-8b, qwen3-32b, ministral-3-14b, gpt-oss-20b, phi-4-mini. No 70B pod: the join is
complete within the ≤32B panel (measured called revealed cells on disk), and large models' order
behaviour is largely predetermined by conjoint position saturation — a scale sentinel adds cost, not
identification; scale is read from the four Gemma 3 sizes plus 27–32B dense models in-panel, reported
per family.

## 8. Predictions (fixed) and what counts as which

**(i) The spinning-arrow view** (presentation parameters dominate content; the critique generalised
from paraphrase to order and opt-out):

- SA1: mean |Δ raw item score| under order/numbering ≥ 0.4 (approaching the paraphrase's 0.61).
- SA2: few or no hard survivors — at most 4 of 17 axes reach R3 in ≥ 2/3 of reliable models; the
  median axis survives in < 1/3.
- SA3: survival adds nothing to validity — the within-model concordance gain among surviving axes
  (A6 permutation) is not significant, and at matched |d| survival rates do not differ between
  revealed-called and non-called axes.
- SA4: the opt-out collapses substantive mass everywhere (pivot triggers; refusal-gated-DK dominant)
  and DK uptake is unstructured across topics.
- SA5: anchor D collapses under L-canonical — the models key on printed numbers, not positions or
  content.

**Counts as spinning-arrow if** SA1 holds **and** SA2 holds **and** SA3 holds: the response paper's
line is then "order and the opt-out spin the survey at least as hard as paraphrase does, and knowing
which topics survive buys no validity — the conjoint remains the only readout that sees these
preferences".

**(ii) The some-topics-stable view** (preferences differ in robustness by topic; the survey is
fixable):

- ST1: the framing-stable four (trans_sports, state_surveillance, voting, drugs — the issues that
  hold direction across focal framings in ≥ 80% of models and have unanimous revealed direction)
  reach R3 in ≥ 2/3 of reliable models, while minimum_wage, guns, agi_limitations, property_tax stall
  at R0–R1; axis-group interaction significant by permutation over axes.
- ST2: the two quantities separate — |Δd| order-range ≪ raw-score order-range (the paired estimand is
  stable where raw answers spin); acquiescence a is high and flat exactly where raw stability is
  spurious.
- ST3: within-model stated-vs-revealed concordance is higher among surviving axes than non-surviving
  at matched |d| (pre-registered one-sided permutation, p < 0.05).
- ST4: DK uptake is topic-structured — lower on axes with called revealed leans (within-model
  permutation): models take the legitimate "no opinion" exactly where they do not act.

**Counts as some-topics-stable if** ST1 holds **and** (ST2 or ST3). Mixed outcomes are expected and
reportable: the table is the result; e.g. "identity/civil-liberties topics survive the machinery,
fiscal/mandate topics spin, and survival predicts which topics the conjoint measures" is the
strongest version; "survival is extremity re-described" (SA2/SA3 with ST2) is the honest null.

## 9. Cost, schedule, ops

- **Pilot:** 1 × RTX PRO 6000 (96 GB), ~45–60 min incl. loads ≈ $2–3.
- **Panel:** 2 × 96 GB pods, one evening. Per-model compute at 3,788 prompts (measured stated-run
  throughputs: 1–4B 60–100 p/s → ~1–3 min; 7–14B 20–48 → ~2–6 min; 27–34B 7–13 → ~5–9 min) is
  dominated by 8–15 min/model load+download: ~12–20 min/model. Pod 1 (ten smaller models) ~3–3.5 h;
  pod 2 (nine larger) ~3.5–4.5 h; ~4 billed hours each = 8 GPU-h ≈ $20 at $2.5/GPU-h.
- **Slack** for one crashed model or re-download: +$10. **Total ≈ $30–40, cap $45.**
- Ops, all inherited: destroy-on-finish pods with deadlines; `guard_p2p` recorded in every meta;
  Gemma 3 eager attention; non-thinking formats for reasoning models; resume keys extended
  (order|numbering|optout) so an interrupted pod resumes; non-finite rows auto-dropped on resume;
  raw trials archived `.jsonl.gz`; analyzer `scripts/pct/orderforce_analyze.py` writes
  `experiments/pct/analysis/orderforce.md` implementing exactly §5–§6, deviations logged as
  amendments.

## 10. Risks and honest limitations

- **Selection into answering** conditions every F1/DK-conditional estimate; F1 is demoted to
  descriptive by design and the pilot decides the opt-out rung's fate before the panel.
- **Pair statements are matched-strength policies, not negations** (the reword experiment showed tone
  moves these models as much as policy): a absorbs wording imbalance; d is directional *under these
  wordings*. Null axes bound the floor; anchors give the acquiescence estimate; the wording-swap
  question stays answered by the existing reword experiment, not re-litigated here.
- **Position-saturated or number-fixated models** may fail R1 for mechanical reasons: reported as a
  per-model diagnostic (position-mass curve, anchor D), with the numbering-limited variant, never
  silently excluded. Scale confound (small models over-represented): per-model and per-family tables
  primary.
- **4 paraphrases per cell** cannot estimate order × paraphrase interaction; the 10-paraphrase
  reference bounds paraphrase sensitivity separately. A cheap 3-model follow-up is noted if the
  interaction is ever needed.
- **One DK wording**, untested for wording sensitivity; reluctance vs no-opinion not separable —
  caveated, not solved. The opt-out's own position/number adjacency is controlled by the opt-out-first
  cell.
- **Cross-host identity gate**: tolerance scaled to what the pilot measures on a same-model
  comparison, since disk rows come from other hosts.
- **Multiplicity**: q = 0.01 within model over 19 axes; empirical null calibration with binomial
  tolerance; flip-proposition negative control must fail.

## 11. What we will NOT claim

- **Not "the model's true preference."** "Survives" means invariant to the tested perturbations —
  these 4 orders, these 2 numberings, these forcing wordings, this one opt-out wording, these 4
  paraphrases. The reword experiment already showed a hostile tone swap reverses 14 of 57 revealed
  leans; stability is manipulation-relative by nature, and the write-up says so.
- **Nothing about unforced free text.** The F1/DK-conditional estimates condition on answer mass that
  exists; refusal-gated is a reported class, not a failure, and where the survey sees nothing we say
  so. Our claims are about the forced and opt-out-answerable subsets.
- **No cross-instrument magnitude claims.** Stated score units and revealed nats are not commensurable
  (the repo's recorded caveat): the offset is discussed by sign consistency only; no "the offset
  shrinks" claim.
- **No pooling that hides model composition.** Pooled topic fractions are secondary; no claims about
  closed models, about where leans come from, or about real voter-assistant behaviour.
- **No ecological claim that scrambled questionnaires are common.** L-canonical numbering is a
  decomposition instrument; comprehension collapse under it is a finding about models, not evidence of
  topic fragility.
- **No per-proposition revealed-preference calls** from 128-trial conjoint cells.
- **No reading of a null result as "models have no preferences"** — only as "the survey cannot see
  them under these presentations", which is the paper's own point, made item-level.

## Amendments

| date | change | reason |
|---|---|---|
| — | — | none yet; pilot outcomes (§4) to be appended before the panel launches |
