# Order-forcing pilot gates (PLAN §4)

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

