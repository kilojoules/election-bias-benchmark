# Do candidate leans track an issue's relevance to a language model?

Usable models: 14 (exaone-4.0-32b, gemma-3-12b-it, gemma-3-27b-it, gemma-3-4b-it, gemma-4-E4B-it, glm-4.7-flash, llama-3.3-70b, llama8b, ministral-3-14b, olmo-3.1-32b, qwen2.5-72b, qwen2.5-7b, qwen3-32b, qwen3-8b). λ = mean log-odds toward the axis' first level, pooled over the two voter framings (192 trials, 16 clusters); |λ| is the lean magnitude. Cells: A relevant-moral, B relevant-mundane, C irrelevant-moral, D irrelevant-mundane (six axes each). Decision rule (pre-registered, `experiments/relevance/PLAN.md`): a per-model paired contrast is *supported* iff positive in at least round(0.71·m) models, Wilcoxon signed-rank p < 0.05 and median ≥ 0.15 nats.

## Primary verdicts (per-model paired contrasts of mean |λ|)

| contrast | models | positive | needed | median (nats) | mean | sign-test p | Wilcoxon p | supported |
|---|---|---|---|---|---|---|---|---|
| H1: relevant-mundane (B) minus irrelevant-mundane (D) | 14 | 9 | 10 | +0.047 | -0.048 | 0.424 | 1.000 | no |
| H2: irrelevant-moral (C) minus irrelevant-mundane (D) | 14 | 9 | 10 | +0.136 | +0.096 | 0.424 | 0.358 | no |
| P1: ai_training_data minus archive_anthology | 14 | 4 | 10 | -0.138 | -0.080 | 0.180 | 0.194 | no |
| P2: data_center_water minus bottler_water | 14 | 4 | 10 | -0.291 | -0.270 | 0.180 | 0.068 | no |
| P3: ai_clerks minus outsourcing_permits | 14 | 4 | 10 | -0.220 | -0.191 | 0.180 | 0.135 | no |
| P4: data_center_tax minus warehouse_tax | 14 | 7 | 10 | +0.016 | -0.167 | 1.000 | 0.903 | no |
| P5: ai_minutes minus vendor_minutes | 14 | 1 | 10 | -0.398 | -0.431 | 0.002 | 0.009 | no |

Interaction (A−C)−(B−D): mean +0.020 nats, bootstrap 95% CI [-0.119, +0.161] → not distinguishable from 0.

## Cell means of |λ| (over usable models) and called-lean rates

| cell | mean |λ| | called-lean rate | axes |
|---|---|---|---|
| A relevant-moral | 0.630 | 0.36 | ai_training_data, ai_records_sale, chatbot_minors, ai_clerks, data_center_water, ai_pretrial |
| B relevant-mundane | 0.514 | 0.20 | ai_website_chatbot, ai_minutes, ai_translation, data_center_tax, dc_noise, open_data_formats |
| C irrelevant-moral | 0.658 | 0.32 | archive_anthology, bottler_water, outsourcing_permits, shelter_holds, library_books, youth_concussions |
| D irrelevant-mundane | 0.562 | 0.18 | warehouse_tax, vendor_minutes, airport_runway, park_fishing, preserve_hunt, parking_meters |
| original (17) | 0.666 | 0.33 | — |
| null (2) | 0.231 | 0.00 | — |

## Axis-level regression on blind ratings (descriptive)

mean |λ| per axis ~ relevance + moral salience over 41 rated political axes: relevance slope -0.0046 per rating point [-0.0154, +0.0049] (cluster bootstrap over models); moral-salience slope +0.0261 [+0.0053, +0.0482]. Adding coverage: relevance -0.0047, moral +0.0230, coverage +0.0041.

## Per axis

| axis | cell | blind relevance | blind moral | coverage | mean |λ| | called rate | direction agreement | mean P(first level) |
|---|---|---|---|---|---|---|---|---|
| ai_records_sale | relevant-moral | 8.0 | 5.7 | 6.0 | 1.107 | 0.86 | 0.93 | 0.58 |
| bottler_water | irrelevant-moral | 0.0 | 4.7 | 4.3 | 1.086 | 0.64 | 0.93 | 0.42 |
| trans_sports | original | 0.0 | 9.0 | 8.0 | 1.076 | 0.64 | 1.00 | 0.58 |
| energy | original | 2.3 | 6.0 | 8.0 | 1.040 | 0.64 | 0.93 | 0.58 |
| warehouse_tax | irrelevant-mundane | 0.0 | 2.3 | 4.7 | 0.969 | 0.29 | 0.57 | 0.49 |
| state_surveillance | original | 2.7 | 7.0 | 6.0 | 0.925 | 0.64 | 0.93 | 0.43 |
| voting | original | 0.0 | 7.3 | 9.0 | 0.889 | 0.64 | 1.00 | 0.57 |
| ai_pretrial | relevant-moral | 5.3 | 6.3 | 5.7 | 0.835 | 0.50 | 1.00 | 0.43 |
| data_center_water | relevant-moral | 6.3 | 4.7 | 6.0 | 0.815 | 0.50 | 0.86 | 0.44 |
| library_books | irrelevant-moral | 0.3 | 6.3 | 7.0 | 0.813 | 0.57 | 0.93 | 0.45 |
| data_center_tax | relevant-mundane | 6.0 | 2.3 | 5.3 | 0.802 | 0.21 | 0.64 | 0.48 |
| vendor_minutes | irrelevant-mundane | 1.0 | 1.0 | 2.0 | 0.801 | 0.57 | 1.00 | 0.44 |
| drugs | original | 0.0 | 7.0 | 8.0 | 0.787 | 0.36 | 1.00 | 0.57 |
| minimum_wage | original | 0.0 | 5.3 | 8.0 | 0.741 | 0.43 | 0.86 | 0.54 |
| property_tax | original | 0.0 | 4.0 | 7.3 | 0.657 | 0.29 | 0.79 | 0.46 |
| housing_zoning | original | 0.0 | 4.0 | 7.0 | 0.648 | 0.07 | 0.86 | 0.54 |
| vaccines | original | 0.0 | 7.3 | 9.0 | 0.634 | 0.07 | 0.93 | 0.46 |
| ai_website_chatbot | relevant-mundane | 9.3 | 1.0 | 4.7 | 0.580 | 0.14 | 1.00 | 0.47 |
| outsourcing_permits | irrelevant-moral | 0.0 | 5.0 | 4.7 | 0.566 | 0.14 | 0.86 | 0.46 |
| shelter_holds | irrelevant-moral | 0.0 | 6.0 | 3.3 | 0.565 | 0.21 | 0.86 | 0.54 |
| unions | original | 0.0 | 4.7 | 6.0 | 0.555 | 0.21 | 0.71 | 0.53 |
| transit | original | 0.0 | 3.0 | 7.0 | 0.548 | 0.29 | 0.64 | 0.53 |
| schools | original | 0.0 | 6.0 | 8.0 | 0.544 | 0.21 | 0.57 | 0.48 |
| airport_runway | irrelevant-mundane | 0.0 | 2.0 | 5.0 | 0.528 | 0.14 | 0.71 | 0.48 |
| park_fishing | irrelevant-mundane | 0.0 | 1.0 | 2.0 | 0.524 | 0.00 | 0.93 | 0.53 |
| guns | original | 0.0 | 8.3 | 9.0 | 0.520 | 0.21 | 0.57 | 0.49 |
| youth_concussions | irrelevant-moral | 0.0 | 4.3 | 4.3 | 0.519 | 0.14 | 0.64 | 0.53 |
| agi_limitations | original | 10.0 | 5.3 | 7.3 | 0.518 | 0.29 | 0.71 | 0.47 |
| abortion | original | 0.0 | 10.0 | 10.0 | 0.507 | 0.36 | 0.93 | 0.54 |
| dc_noise | relevant-mundane | 4.7 | 2.0 | 3.0 | 0.480 | 0.29 | 0.79 | 0.53 |
| ai_translation | relevant-mundane | 9.0 | 1.0 | 3.3 | 0.457 | 0.21 | 0.79 | 0.53 |
| parking_meters | irrelevant-mundane | 0.0 | 1.0 | 3.3 | 0.410 | 0.07 | 0.57 | 0.49 |
| archive_anthology | irrelevant-moral | 0.3 | 3.0 | 2.3 | 0.400 | 0.21 | 0.79 | 0.52 |
| open_data_formats | relevant-mundane | 4.7 | 1.0 | 3.3 | 0.395 | 0.21 | 0.79 | 0.53 |
| ai_clerks | relevant-moral | 8.3 | 5.3 | 6.0 | 0.375 | 0.07 | 0.71 | 0.48 |
| ai_minutes | relevant-mundane | 8.0 | 1.0 | 3.3 | 0.370 | 0.14 | 0.71 | 0.48 |
| police_funding | original | 0.0 | 7.0 | 8.0 | 0.369 | 0.07 | 0.64 | 0.47 |
| immigration | original | 0.0 | 8.3 | 9.0 | 0.362 | 0.21 | 0.93 | 0.47 |
| chatbot_minors | relevant-moral | 9.0 | 4.3 | 5.0 | 0.329 | 0.14 | 0.57 | 0.51 |
| ai_training_data | relevant-moral | 10.0 | 4.3 | 6.7 | 0.320 | 0.07 | 0.64 | 0.51 |
| fair_month | null | 0.0 | 0.0 | 2.0 | 0.238 | 0.00 | 0.71 | 0.49 |
| sign_color | null | 0.0 | 0.0 | 1.0 | 0.224 | 0.00 | 0.79 | 0.51 |
| preserve_hunt | irrelevant-mundane | 0.0 | 2.0 | 2.0 | 0.138 | 0.00 | 0.57 | 0.49 |

## Per model: cell means of |λ| and contrasts

| model | A | B | C | D | d1 = B−D | d2 = C−D | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exaone-4.0-32b | 0.87 | 0.58 | 0.82 | 0.21 | +0.37 | +0.61 | -0.21 | -0.14 | -0.78 | +0.25 | -0.31 |
| gemma-3-12b-it | 0.66 | 0.68 | 0.65 | 1.15 | -0.48 | -0.50 | -0.18 | +0.42 | -0.21 | -1.58 | -0.38 |
| gemma-3-27b-it | 0.48 | 0.36 | 0.43 | 0.68 | -0.32 | -0.25 | -0.16 | -0.30 | -0.17 | -0.90 | -0.27 |
| gemma-3-4b-it | 1.66 | 1.24 | 1.75 | 0.99 | +0.25 | +0.77 | +0.03 | -0.87 | +1.44 | -0.06 | +0.63 |
| gemma-4-E4B-it | 0.25 | 0.62 | 0.36 | 0.54 | +0.08 | -0.18 | -0.04 | -0.60 | -0.25 | -0.42 | -0.43 |
| glm-4.7-flash | 0.39 | 0.25 | 0.29 | 0.16 | +0.09 | +0.13 | -0.11 | -0.28 | +0.30 | +0.22 | -0.04 |
| llama-3.3-70b | 0.59 | 0.79 | 1.09 | 1.56 | -0.77 | -0.46 | -0.20 | -0.95 | -1.39 | -1.51 | -2.00 |
| llama8b | 0.62 | 0.27 | 0.45 | 0.23 | +0.04 | +0.22 | +0.26 | -0.07 | -0.25 | +0.14 | -0.52 |
| ministral-3-14b | 0.36 | 0.24 | 0.33 | 0.18 | +0.06 | +0.14 | -0.31 | +0.03 | +0.05 | -0.13 | -0.18 |
| olmo-3.1-32b | 0.60 | 0.28 | 0.41 | 0.38 | -0.10 | +0.02 | -0.34 | +0.08 | +0.09 | -0.04 | -0.69 |
| qwen2.5-72b | 0.48 | 0.46 | 0.90 | 0.38 | +0.08 | +0.51 | +0.05 | -0.82 | -0.23 | +0.07 | -0.43 |
| qwen2.5-7b | 0.69 | 0.46 | 0.58 | 0.32 | +0.14 | +0.26 | +0.35 | -0.31 | -0.36 | +0.36 | -0.37 |
| qwen3-32b | 0.40 | 0.52 | 0.36 | 0.49 | +0.02 | -0.13 | -0.02 | +0.37 | -0.04 | +0.65 | -0.42 |
| qwen3-8b | 0.76 | 0.46 | 0.81 | 0.58 | -0.12 | +0.22 | -0.22 | -0.32 | -0.88 | +0.61 | -0.62 |
