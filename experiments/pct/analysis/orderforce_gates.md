# Order-forcing pilot gates (PLAN §4)

## gemma-3-12b-it  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.06, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.99 (F1 0.00), median P(opt-out) 0.00 → opt-out gate stands
- anchors: D reference 1.76 (OK), D canonical median 1.77 (OK)
- validity (valid mass) F1 0.00 → F3 1.00 → F5 1.00
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.88 (24/24 named; {'number': 24}) FAIL; reversed|canonical|3|none: 0.96 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 0.92 (24/24 named; {'label': 10, 'number': 14}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 1.02 | 0.98 |
| paper|positional|1|none | 0.37 | 1.63 |
| paper|positional|2|first | 1.24 | 0.56 |
| paper|positional|2|last | 1.88 | 0.12 |
| paper|positional|3|none | 1.76 | 0.14 |
| paper|positional|5|none | 1.88 | 0.09 |
| reversed|canonical|3|none | 1.97 | 0.02 |
| reversed|positional|2|last | 1.89 | 0.11 |
| reversed|positional|3|none | 1.92 | 0.08 |
| scrambled2|canonical|3|none | 0.81 | -1.16 |
| scrambled2|positional|2|last | 0.71 | -0.69 |
| scrambled2|positional|3|none | 1.75 | -0.08 |
| scrambled|canonical|3|none | 1.77 | -0.08 |
| scrambled|positional|2|last | 1.09 | -0.08 |
| scrambled|positional|3|none | 1.41 | -0.09 |

## gemma-3-1b-it  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.02, max 0.26, prompts with |Δ| > 0.10: 0.05 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 1.00 (F1 0.00), median P(opt-out) 0.00 → opt-out gate stands
- anchors: D reference -0.27 (FAIL), D canonical median 0.10 (numbering-limited)
- validity (valid mass) F1 0.00 → F3 1.00 → F5 0.98
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.96 (24/24 named; {'number': 9, 'label': 15}) OK; reversed|canonical|3|none: 0.96 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 0.96 (24/24 named; {'number': 20, 'label': 4}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | -0.18 | 0.96 |
| paper|positional|1|none | -0.37 | 0.82 |
| paper|positional|2|first | -0.69 | 0.19 |
| paper|positional|2|last | -0.14 | 0.98 |
| paper|positional|3|none | -0.27 | 1.10 |
| paper|positional|5|none | -0.21 | 1.23 |
| reversed|canonical|3|none | 0.10 | -1.07 |
| reversed|positional|2|last | 0.57 | -0.83 |
| reversed|positional|3|none | 0.71 | -0.69 |
| scrambled2|canonical|3|none | -0.52 | -0.65 |
| scrambled2|positional|2|last | 0.02 | 0.36 |
| scrambled2|positional|3|none | 0.09 | 0.42 |
| scrambled|canonical|3|none | 0.69 | -0.35 |
| scrambled|positional|2|last | 1.29 | -0.30 |
| scrambled|positional|3|none | 1.03 | -0.04 |

## gemma-3-27b-it  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.00, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 1.00 (F1 0.00), median P(opt-out) 0.00 → opt-out gate stands
- anchors: D reference 1.93 (OK), D canonical median 2.00 (OK)
- validity (valid mass) F1 0.00 → F3 0.99 → F5 1.00
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 1.00 (24/24 named; {'number': 10, 'label': 14}) OK; reversed|canonical|3|none: 0.96 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 1.00 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 1.72 | 0.28 |
| paper|positional|1|none | 1.79 | 0.21 |
| paper|positional|2|first | 1.71 | 0.29 |
| paper|positional|2|last | 1.98 | 0.02 |
| paper|positional|3|none | 1.93 | 0.07 |
| paper|positional|5|none | 1.75 | 0.25 |
| reversed|canonical|3|none | 2.00 | 0.00 |
| reversed|positional|2|last | 1.90 | -0.05 |
| reversed|positional|3|none | 1.86 | -0.02 |
| scrambled2|canonical|3|none | 2.00 | 0.00 |
| scrambled2|positional|2|last | 1.28 | -0.28 |
| scrambled2|positional|3|none | 1.39 | -0.34 |
| scrambled|canonical|3|none | 1.92 | 0.08 |
| scrambled|positional|2|last | 1.27 | -0.24 |
| scrambled|positional|3|none | 1.27 | 0.03 |

## gemma-3-4b-it  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.01, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 1.00 (F1 0.00), median P(opt-out) 0.00 → opt-out gate stands
- anchors: D reference 1.07 (OK), D canonical median 0.98 (OK)
- validity (valid mass) F1 0.00 → F3 0.99 → F5 1.00
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.92 (24/24 named; {'number': 19, 'label': 5}) OK; reversed|canonical|3|none: 1.00 (24/24 named; {'number': 21, 'label': 3}) OK; paper|positional|2|last: 1.00 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.94 | 0.23 |
| paper|positional|1|none | 1.11 | 0.16 |
| paper|positional|2|first | 1.29 | -0.29 |
| paper|positional|2|last | 1.13 | -0.21 |
| paper|positional|3|none | 1.07 | -0.20 |
| paper|positional|5|none | 1.21 | -0.21 |
| reversed|canonical|3|none | 1.21 | 0.22 |
| reversed|positional|2|last | 1.26 | 0.55 |
| reversed|positional|3|none | 1.02 | 0.80 |
| scrambled2|canonical|3|none | 0.14 | -1.69 |
| scrambled2|positional|2|last | 0.89 | -0.61 |
| scrambled2|positional|3|none | 0.65 | -0.80 |
| scrambled|canonical|3|none | 0.98 | 0.99 |
| scrambled|positional|2|last | 0.81 | 0.28 |
| scrambled|positional|3|none | 0.61 | 0.85 |

## gemma-4-E4B-it  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.06, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.11 (F1 0.00), median P(opt-out) 0.90 → PIVOT: refusal-gated-DK
- anchors: D reference 1.48 (OK), D canonical median 1.81 (OK)
- validity (valid mass) F1 0.00 → F3 0.86 → F5 0.62
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.67 (18/24 named; {'other': 6, 'number': 14, 'label-inline': 4}) FAIL; reversed|canonical|3|none: 0.80 (15/24 named; {'other': 9, 'number': 10, 'label-inline': 5}) FAIL; paper|positional|2|last: 0.96 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.41 | -1.26 |
| paper|positional|1|none | 0.36 | -1.15 |
| paper|positional|2|first | 1.49 | -0.48 |
| paper|positional|2|last | 1.67 | -0.21 |
| paper|positional|3|none | 1.48 | -0.30 |
| paper|positional|5|none | 1.29 | -0.27 |
| reversed|canonical|3|none | 1.45 | -0.47 |
| reversed|positional|2|last | 1.40 | -0.53 |
| reversed|positional|3|none | 1.46 | -0.42 |
| scrambled2|canonical|3|none | 1.87 | -0.12 |
| scrambled2|positional|2|last | 1.61 | -0.07 |
| scrambled2|positional|3|none | 1.71 | 0.05 |
| scrambled|canonical|3|none | 1.81 | -0.01 |
| scrambled|positional|2|last | 1.67 | -0.22 |
| scrambled|positional|3|none | 1.71 | -0.17 |

## gpt-oss-20b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.01, max 0.08, prompts with |Δ| > 0.10: 0.02 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.89 (F1 0.00), median P(opt-out) 0.11 → opt-out gate stands
- anchors: D reference 1.08 (OK), D canonical median 0.82 (OK)
- validity (valid mass) F1 0.00 → F3 0.99 → F5 0.93
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.96 (24/24 named; {'number': 24}) OK; reversed|canonical|3|none: 1.00 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 0.96 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.60 | -1.01 |
| paper|positional|1|none | 0.90 | -0.73 |
| paper|positional|2|first | 0.82 | 0.43 |
| paper|positional|2|last | 0.98 | 0.49 |
| paper|positional|3|none | 1.08 | 0.22 |
| paper|positional|5|none | 1.10 | 0.18 |
| reversed|canonical|3|none | 0.73 | 0.23 |
| reversed|positional|2|last | 1.46 | 0.09 |
| reversed|positional|3|none | 1.49 | 0.11 |
| scrambled2|canonical|3|none | 0.82 | 0.27 |
| scrambled2|positional|2|last | 0.83 | 0.04 |
| scrambled2|positional|3|none | 0.82 | 0.23 |
| scrambled|canonical|3|none | 0.86 | 0.41 |
| scrambled|positional|2|last | 1.08 | 0.11 |
| scrambled|positional|3|none | 1.10 | -0.03 |

## granite-4.2-8b  (3788 rows)

- identity vs arm A: 0 items, mean |Δ| —, max —, prompts with |Δ| > 0.10: — → n/a
- DK cell (O1, F2, opt-out last): median substantive mass 0.99 (F1 0.00), median P(opt-out) 0.00 → opt-out gate stands
- anchors: D reference 1.70 (OK), D canonical median 0.40 (numbering-limited)
- validity (valid mass) F1 0.01 → F3 0.96 → F5 0.58
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.95 (19/24 named; {'number': 19, 'other': 5}) OK; reversed|canonical|3|none: 0.91 (23/24 named; {'number': 23, 'other': 1}) OK; paper|positional|2|last: 1.00 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 1.11 | -0.24 |
| paper|positional|1|none | 1.02 | -0.29 |
| paper|positional|2|first | 1.36 | -0.62 |
| paper|positional|2|last | 1.66 | 0.01 |
| paper|positional|3|none | 1.70 | 0.06 |
| paper|positional|5|none | 1.69 | -0.07 |
| reversed|canonical|3|none | 0.71 | -0.16 |
| reversed|positional|2|last | 1.91 | -0.08 |
| reversed|positional|3|none | 1.89 | -0.10 |
| scrambled2|canonical|3|none | 0.40 | -0.79 |
| scrambled2|positional|2|last | 1.38 | -0.17 |
| scrambled2|positional|3|none | 1.13 | -0.12 |
| scrambled|canonical|3|none | 0.38 | -0.09 |
| scrambled|positional|2|last | 1.32 | -0.38 |
| scrambled|positional|3|none | 1.44 | -0.32 |

## llama-3.2-3b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.01, max 0.04, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.93 (F1 0.00), median P(opt-out) 0.01 → opt-out gate stands
- anchors: D reference 1.13 (OK), D canonical median 0.88 (OK)
- validity (valid mass) F1 0.00 → F3 0.85 → F5 0.63
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.96 (23/24 named; {'number': 23, 'other': 1}) OK; reversed|canonical|3|none: 0.95 (20/24 named; {'number': 20, 'other': 4}) OK; paper|positional|2|last: 1.00 (6/24 named; {'other': 18, 'number': 6}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.82 | -0.11 |
| paper|positional|1|none | 1.05 | -0.22 |
| paper|positional|2|first | 0.66 | 0.14 |
| paper|positional|2|last | 0.92 | 0.19 |
| paper|positional|3|none | 1.13 | -0.25 |
| paper|positional|5|none | 1.08 | -0.17 |
| reversed|canonical|3|none | 0.54 | -0.24 |
| reversed|positional|2|last | 1.00 | -0.25 |
| reversed|positional|3|none | 1.03 | -0.14 |
| scrambled2|canonical|3|none | 0.88 | -0.13 |
| scrambled2|positional|2|last | 1.00 | -0.55 |
| scrambled2|positional|3|none | 1.15 | -0.40 |
| scrambled|canonical|3|none | 1.05 | 0.41 |
| scrambled|positional|2|last | 1.17 | 0.26 |
| scrambled|positional|3|none | 1.22 | 0.13 |

## llama8b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.01, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.95 (F1 0.00), median P(opt-out) 0.05 → opt-out gate stands
- anchors: D reference 0.80 (OK), D canonical median 0.86 (OK)
- validity (valid mass) F1 0.01 → F3 1.00 → F5 0.98
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 1.00 (24/24 named; {'number': 24}) OK; reversed|canonical|3|none: 0.96 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 0.96 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.95 | 0.29 |
| paper|positional|1|none | 0.94 | 0.37 |
| paper|positional|2|first | 0.93 | -0.16 |
| paper|positional|2|last | 0.74 | 0.47 |
| paper|positional|3|none | 0.80 | 0.31 |
| paper|positional|5|none | 0.77 | 0.25 |
| reversed|canonical|3|none | 1.00 | -0.22 |
| reversed|positional|2|last | 1.16 | -0.53 |
| reversed|positional|3|none | 1.20 | -0.40 |
| scrambled2|canonical|3|none | 0.59 | 0.12 |
| scrambled2|positional|2|last | 1.29 | -0.38 |
| scrambled2|positional|3|none | 1.19 | -0.53 |
| scrambled|canonical|3|none | 0.86 | 0.04 |
| scrambled|positional|2|last | 0.94 | 0.82 |
| scrambled|positional|3|none | 1.04 | 0.72 |

## ministral-3-14b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.02, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.45 (F1 0.16), median P(opt-out) 0.23 → opt-out gate stands
- anchors: D reference 0.74 (OK), D canonical median 0.35 (numbering-limited)
- validity (valid mass) F1 0.16 → F3 0.52 → F5 0.44
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.92 (24/24 named; {'number': 24}) OK; reversed|canonical|3|none: 0.96 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 0.96 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.36 | -0.06 |
| paper|positional|1|none | 0.41 | 0.06 |
| paper|positional|2|first | 0.31 | -0.30 |
| paper|positional|2|last | 0.49 | -0.03 |
| paper|positional|3|none | 0.74 | 0.08 |
| paper|positional|5|none | 0.82 | 0.36 |
| reversed|canonical|3|none | 0.35 | -0.16 |
| reversed|positional|2|last | 0.77 | -0.20 |
| reversed|positional|3|none | 0.96 | -0.37 |
| scrambled2|canonical|3|none | 0.39 | -0.20 |
| scrambled2|positional|2|last | 0.45 | -0.05 |
| scrambled2|positional|3|none | 0.36 | -0.11 |
| scrambled|canonical|3|none | 0.25 | 0.19 |
| scrambled|positional|2|last | 0.36 | 0.17 |
| scrambled|positional|3|none | 0.37 | 0.10 |

## ministral-3-8b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.01, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.38 (F1 0.00), median P(opt-out) 0.22 → opt-out gate stands
- anchors: D reference 0.48 (FAIL), D canonical median 0.43 (numbering-limited)
- validity (valid mass) F1 0.01 → F3 0.67 → F5 0.55
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.92 (24/24 named; {'number': 22, 'label': 2}) OK; reversed|canonical|3|none: 0.83 (24/24 named; {'number': 23, 'label': 1}) FAIL; paper|positional|2|last: 0.46 (24/24 named; {'number': 6, 'label': 18}) FAIL

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.88 | -0.18 |
| paper|positional|1|none | 0.90 | -0.15 |
| paper|positional|2|first | 0.83 | 0.16 |
| paper|positional|2|last | 0.72 | 0.44 |
| paper|positional|3|none | 0.48 | 0.57 |
| paper|positional|5|none | 0.57 | 0.78 |
| reversed|canonical|3|none | 0.43 | 0.22 |
| reversed|positional|2|last | 0.93 | -0.13 |
| reversed|positional|3|none | 0.77 | -0.32 |
| scrambled2|canonical|3|none | 0.24 | 0.37 |
| scrambled2|positional|2|last | 1.13 | -0.12 |
| scrambled2|positional|3|none | 0.49 | 0.02 |
| scrambled|canonical|3|none | 0.43 | 0.32 |
| scrambled|positional|2|last | 0.97 | 0.12 |
| scrambled|positional|3|none | 0.35 | 0.02 |

## olmo-3.1-32b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.01, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.18 (F1 0.00), median P(opt-out) 0.76 → PIVOT: refusal-gated-DK
- anchors: D reference 1.64 (OK), D canonical median 1.49 (OK)
- validity (valid mass) F1 0.00 → F3 0.71 → F5 0.72
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 1.00 (19/24 named; {'other': 5, 'number': 15, 'label': 4}) OK; reversed|canonical|3|none: 1.00 (19/24 named; {'number': 19, 'other': 5}) OK; paper|positional|2|last: 0.96 (24/24 named; {'label': 20, 'number': 4}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 1.23 | 0.31 |
| paper|positional|1|none | 1.52 | -0.10 |
| paper|positional|2|first | 1.50 | 0.16 |
| paper|positional|2|last | 1.44 | 0.30 |
| paper|positional|3|none | 1.64 | 0.09 |
| paper|positional|5|none | 1.42 | 0.28 |
| reversed|canonical|3|none | 1.79 | 0.06 |
| reversed|positional|2|last | 1.51 | -0.23 |
| reversed|positional|3|none | 1.71 | -0.25 |
| scrambled2|canonical|3|none | 1.49 | -0.41 |
| scrambled2|positional|2|last | 1.25 | 0.07 |
| scrambled2|positional|3|none | 1.49 | 0.27 |
| scrambled|canonical|3|none | 1.22 | 0.69 |
| scrambled|positional|2|last | 1.27 | -0.19 |
| scrambled|positional|3|none | 1.27 | -0.04 |

## phi-4-mini  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.01, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.89 (F1 0.16), median P(opt-out) 0.04 → opt-out gate stands
- anchors: D reference 1.25 (OK), D canonical median 1.02 (OK)
- validity (valid mass) F1 0.21 → F3 0.81 → F5 0.78
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.83 (24/24 named; {'number': 24}) FAIL; reversed|canonical|3|none: 0.92 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 0.79 (24/24 named; {'number': 24}) FAIL

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 1.08 | 0.52 |
| paper|positional|1|none | 1.03 | 0.37 |
| paper|positional|2|first | 1.22 | -0.23 |
| paper|positional|2|last | 1.32 | -0.05 |
| paper|positional|3|none | 1.25 | -0.17 |
| paper|positional|5|none | 1.21 | -0.10 |
| reversed|canonical|3|none | 1.02 | -0.89 |
| reversed|positional|2|last | 1.37 | -0.48 |
| reversed|positional|3|none | 1.28 | -0.31 |
| scrambled2|canonical|3|none | 0.31 | 0.39 |
| scrambled2|positional|2|last | 0.69 | 0.31 |
| scrambled2|positional|3|none | 0.84 | 0.17 |
| scrambled|canonical|3|none | 1.17 | -0.26 |
| scrambled|positional|2|last | 1.05 | -0.10 |
| scrambled|positional|3|none | 1.09 | -0.13 |

## phi-4  (3788 rows)

- identity vs arm A: 0 items, mean |Δ| —, max —, prompts with |Δ| > 0.10: — → n/a
- DK cell (O1, F2, opt-out last): median substantive mass 0.94 (F1 0.00), median P(opt-out) 0.00 → opt-out gate stands
- anchors: D reference 1.80 (OK), D canonical median 1.31 (OK)
- validity (valid mass) F1 0.00 → F3 0.68 → F5 0.62
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 1.00 (11/24 named; {'other': 13, 'number': 11}) OK; reversed|canonical|3|none: 1.00 (5/24 named; {'other': 19, 'number': 5}) OK; paper|positional|2|last: 1.00 (17/24 named; {'number': 17, 'other': 7}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 1.45 | -0.53 |
| paper|positional|1|none | 1.33 | -0.66 |
| paper|positional|2|first | 1.67 | -0.33 |
| paper|positional|2|last | 1.83 | 0.07 |
| paper|positional|3|none | 1.80 | -0.02 |
| paper|positional|5|none | 1.75 | -0.07 |
| reversed|canonical|3|none | 1.21 | -0.32 |
| reversed|positional|2|last | 1.56 | -0.44 |
| reversed|positional|3|none | 1.67 | -0.32 |
| scrambled2|canonical|3|none | 1.43 | -0.24 |
| scrambled2|positional|2|last | 1.20 | 0.00 |
| scrambled2|positional|3|none | 1.34 | 0.31 |
| scrambled|canonical|3|none | 1.31 | 0.21 |
| scrambled|positional|2|last | 1.53 | -0.03 |
| scrambled|positional|3|none | 1.58 | -0.34 |

## qwen2.5-7b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.00, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.98 (F1 0.00), median P(opt-out) 0.00 → opt-out gate stands
- anchors: D reference 1.46 (OK), D canonical median 0.83 (OK)
- validity (valid mass) F1 0.00 → F3 0.98 → F5 0.97
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.75 (24/24 named; {'label': 21, 'number': 3}) FAIL; reversed|canonical|3|none: 0.88 (24/24 named; {'number': 23, 'label': 1}) FAIL; paper|positional|2|last: 1.00 (24/24 named; {'number': 22, 'label': 2}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 1.62 | 0.25 |
| paper|positional|1|none | 1.34 | 0.61 |
| paper|positional|2|first | 1.51 | -0.49 |
| paper|positional|2|last | 1.38 | -0.42 |
| paper|positional|3|none | 1.46 | -0.48 |
| paper|positional|5|none | 1.72 | -0.13 |
| reversed|canonical|3|none | 1.06 | -0.46 |
| reversed|positional|2|last | 1.26 | -0.26 |
| reversed|positional|3|none | 1.18 | -0.09 |
| scrambled2|canonical|3|none | 0.83 | -1.12 |
| scrambled2|positional|2|last | 1.08 | -0.62 |
| scrambled2|positional|3|none | 0.69 | -1.05 |
| scrambled|canonical|3|none | 0.17 | 0.64 |
| scrambled|positional|2|last | 1.59 | 0.14 |
| scrambled|positional|3|none | 1.56 | -0.23 |

## qwen3-32b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.03, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.86 (F1 0.03), median P(opt-out) 0.01 → opt-out gate stands
- anchors: D reference 1.82 (OK), D canonical median 1.75 (OK)
- validity (valid mass) F1 0.02 → F3 0.82 → F5 0.82
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.96 (24/24 named; {'number': 19, 'label': 5}) OK; reversed|canonical|3|none: 1.00 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 1.00 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 1.21 | 0.64 |
| paper|positional|1|none | 1.46 | 0.46 |
| paper|positional|2|first | 1.51 | -0.27 |
| paper|positional|2|last | 1.53 | 0.37 |
| paper|positional|3|none | 1.82 | 0.13 |
| paper|positional|5|none | 1.79 | 0.20 |
| reversed|canonical|3|none | 1.80 | 0.17 |
| reversed|positional|2|last | 1.48 | -0.10 |
| reversed|positional|3|none | 1.79 | -0.06 |
| scrambled2|canonical|3|none | 1.75 | 0.22 |
| scrambled2|positional|2|last | 1.58 | 0.10 |
| scrambled2|positional|3|none | 1.77 | 0.06 |
| scrambled|canonical|3|none | 0.96 | 0.14 |
| scrambled|positional|2|last | 1.75 | 0.21 |
| scrambled|positional|3|none | 1.84 | 0.11 |

## qwen3-8b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.04, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 1.00 (F1 0.00), median P(opt-out) 0.00 → opt-out gate stands
- anchors: D reference 0.72 (OK), D canonical median 0.47 (numbering-limited)
- validity (valid mass) F1 0.00 → F3 1.00 → F5 1.00
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 1.00 (24/24 named; {'number': 24}) OK; reversed|canonical|3|none: 0.96 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 0.96 (24/24 named; {'number': 24}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.30 | 0.65 |
| paper|positional|1|none | 0.37 | 0.62 |
| paper|positional|2|first | 1.09 | -0.09 |
| paper|positional|2|last | 0.81 | 0.22 |
| paper|positional|3|none | 0.72 | 0.29 |
| paper|positional|5|none | 0.51 | 0.50 |
| reversed|canonical|3|none | 0.47 | -0.57 |
| reversed|positional|2|last | 0.79 | 0.21 |
| reversed|positional|3|none | 0.66 | 0.34 |
| scrambled2|canonical|3|none | 0.96 | -0.12 |
| scrambled2|positional|2|last | 0.97 | 0.03 |
| scrambled2|positional|3|none | 1.13 | 0.33 |
| scrambled|canonical|3|none | 0.35 | 1.05 |
| scrambled|positional|2|last | 1.32 | -0.00 |
| scrambled|positional|3|none | 1.18 | 0.17 |

## qwen3.5-9b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.02, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.60 (F1 0.00), median P(opt-out) 0.34 → opt-out gate stands
- anchors: D reference 1.21 (OK), D canonical median 0.92 (OK)
- validity (valid mass) F1 0.01 → F3 0.92 → F5 0.82
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 0.88 (24/24 named; {'number': 24}) FAIL; reversed|canonical|3|none: 0.87 (23/24 named; {'number': 23, 'other': 1}) FAIL; paper|positional|2|last: 1.00 (24/24 named; {'number': 23, 'label': 1}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.68 | 1.04 |
| paper|positional|1|none | 0.78 | 1.05 |
| paper|positional|2|first | 1.18 | -0.01 |
| paper|positional|2|last | 1.17 | 0.23 |
| paper|positional|3|none | 1.21 | 0.22 |
| paper|positional|5|none | 1.30 | 0.27 |
| reversed|canonical|3|none | 0.75 | 0.04 |
| reversed|positional|2|last | 1.28 | 0.04 |
| reversed|positional|3|none | 1.39 | -0.06 |
| scrambled2|canonical|3|none | 0.95 | 0.21 |
| scrambled2|positional|2|last | 1.29 | 0.11 |
| scrambled2|positional|3|none | 1.14 | 0.16 |
| scrambled|canonical|3|none | 0.92 | 0.42 |
| scrambled|positional|2|last | 1.30 | 0.11 |
| scrambled|positional|3|none | 1.22 | 0.02 |

## smollm3-3b  (3788 rows)

- identity vs arm A: 96 items, mean |Δ| 0.00, max 0.03, prompts with |Δ| > 0.10: 0.00 → OK
- DK cell (O1, F2, opt-out last): median substantive mass 0.77 (F1 0.01), median P(opt-out) 0.01 → opt-out gate stands
- anchors: D reference 0.99 (OK), D canonical median 0.85 (OK)
- validity (valid mass) F1 0.05 → F3 0.84 → F5 0.74
- decode vs logit argmax (named decodes): scrambled|positional|3|none: 1.00 (24/24 named; {'number': 24}) OK; reversed|canonical|3|none: 1.00 (24/24 named; {'number': 24}) OK; paper|positional|2|last: 0.96 (24/24 named; {'number': 23, 'label': 1}) OK

| anchors cell | D | A |
|---|---|---|
| paper|positional|1|last | 0.61 | 0.72 |
| paper|positional|1|none | 0.73 | 0.53 |
| paper|positional|2|first | 0.72 | 0.29 |
| paper|positional|2|last | 0.94 | 0.32 |
| paper|positional|3|none | 0.99 | 0.32 |
| paper|positional|5|none | 1.02 | 0.25 |
| reversed|canonical|3|none | 0.99 | -0.08 |
| reversed|positional|2|last | 0.95 | -0.37 |
| reversed|positional|3|none | 1.09 | -0.29 |
| scrambled2|canonical|3|none | 0.50 | 0.21 |
| scrambled2|positional|2|last | 0.72 | 0.09 |
| scrambled2|positional|3|none | 0.73 | 0.12 |
| scrambled|canonical|3|none | 0.85 | 0.54 |
| scrambled|positional|2|last | 0.57 | -0.29 |
| scrambled|positional|3|none | 0.67 | -0.23 |

