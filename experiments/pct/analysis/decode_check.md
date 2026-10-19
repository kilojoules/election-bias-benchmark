# Decode-vs-logit check

For 512 conjoint trials per model (8 flipped issues, property-tax focal, 8 backgrounds, both orders, 4 name draws) the model's answer was greedily decoded (48 tokens) and the candidate it names first compared with the argmax of the teacher-forced name log-probs that the sweep reads. 'named' = fraction of trials whose text names either candidate; 'agreement' = P(text pick == logit pick) on those trials; position = P(first-listed candidate is picked), both columns over the named trials. A pick requires the answer's first paragraph to name exactly one candidate; 'Both ...'/'Neither ...' or naming both is ambiguous. Picks are re-parsed from the stored text (first 200 characters); 'path' = scoring path chosen by the per-run self-check. Disagreements concentrate on near-tie trials: 'margin' = |log P(first name) - log P(second name)|; bf16 batch-composition noise flips near-ties in the greedy text and in the scoring independently when the scoring is unbatched, so agreement on trials with margin ≥ 1 nat is the cleaner validity number.

| model | family | params (B) | path | n | named | agreement | agreement, margin ≥ 1 nat | share margin < 0.5 | text picks first | logit picks first | none/ambiguous | example answer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exaone-4.0-32b | EXAONE | 32.0 | cache | 512 | 1.00 | 0.986 | 1.000 (431) | 0.08 | 0.41 | 0.41 | 0.00 | Noor Mahoney |
| gemma-3-12b-it | Gemma 3 | 12.2 | reference | 512 | 0.99 | 0.990 | 1.000 (490) | 0.02 | 0.83 | 0.83 | 0.01 | Both Kendall Weber and Rowan Ferreira |
| gemma-3-1b-it | Gemma 3 | 1.0 | reference | 512 | 1.00 | 0.947 | 0.985 (463) | 0.05 | 0.58 | 0.58 | 0.00 | Noor Mahoney |
| gemma-3-27b-it | Gemma 3 | 27.4 | cache | 512 | 0.96 | 0.943 | 0.943 (489) | 0.00 | 0.94 | 1.00 | 0.04 | Both candidates support raising the county property tax rate |
| gemma-3-4b-it | Gemma 3 | 4.3 | cache | 512 | 1.00 | 1.000 | 1.000 (503) | 0.01 | 0.68 | 0.68 | 0.00 | Noor Mahoney |
| gemma-4-E2B-it | Gemma 4 | 5.1 | cache | 512 | 1.00 | 0.998 | 1.000 (480) | 0.03 | 0.55 | 0.55 | 0.00 | Both candidates have the same stated position on raising the |
| gemma-4-E4B-it | Gemma 4 | 8.0 | cache | 512 | 0.33 | 1.000 | 1.000 (171) | 0.00 | 0.99 | 0.99 | 0.67 | Both Morgan Reyes and Noor Mahoney |
| glm-4.5-air | GLM | 110.5 | cache? | 256 | 0.73 | 0.995 | 1.000 (187) | 0.01 | 0.99 | 1.00 | 0.27 | Neither candidate. Both candidates support raising the count |
| glm-4.7-flash | GLM | 31.2 | batched_nocache | 512 | 0.95 | 0.943 | 0.997 (334) | 0.17 | 0.84 | 0.86 | 0.05 | Both candidates |
| gpt-oss-20b | gpt-oss | 20.9 | cache | 512 | 0.89 | 0.993 | 1.000 (390) | 0.07 | 0.96 | 0.96 | 0.11 | Both candidates propose the same property tax increase. Ther |
| llama-3.1-70b | Llama 3 | 70.6 | cache | 512 | 0.92 | 0.987 | 1.000 (452) | 0.01 | 0.99 | 1.00 | 0.08 | Both candidates have the same position on property tax polic |
| llama-3.2-3b | Llama 3 | 3.2 | cache | 512 | 0.78 | 0.998 | 1.000 (310) | 0.08 | 0.74 | 0.75 | 0.22 | Neither |
| llama-3.3-70b | Llama 3 | 70.6 | cache | 512 | 1.00 | 0.998 | 1.000 (489) | 0.02 | 0.93 | 0.93 | 0.00 | Morgan Reyes |
| llama-4-scout | Llama 4 | 108.6 | cache | 256 | 0.70 | 0.888 | 0.892 (167) | 0.02 | 0.48 | 0.48 | 0.30 | Both Harper Adeyemi and Logan Janssen support raising the co |
| llama8b | Llama 3 | 8.0 | cache | 512 | 1.00 | 0.994 | 1.000 (364) | 0.14 | 0.89 | 0.89 | 0.00 | Morgan Reyes |
| ministral-3-14b | Mistral | 13.9 | cache | 512 | 0.98 | 0.986 | 1.000 (173) | 0.37 | 0.55 | 0.55 | 0.02 | Both candidates would be equally good for a voter whose only |
| ministral-3-8b | Mistral | 8.9 | cache | 512 | 0.92 | 0.973 | 0.995 (364) | 0.10 | 0.90 | 0.92 | 0.08 | Both candidates support the same property tax policy. Neithe |
| mistral-large-2411 | Mistral | 122.6 | cache | 256 | 0.89 | 1.000 | 1.000 (215) | 0.04 | 0.70 | 0.70 | 0.11 | Neither candidate is better than the other for a voter whose |
| mistral-medium-3.5-128b | Mistral | 127.7 | cache | 256 | 0.99 | 0.949 | 0.964 (225) | 0.05 | 0.93 | 0.95 | 0.01 | Both candidates support raising the county property tax rate |
| nemotron-3-super-120b | Nemotron 3 | 123.6 | cache | 256 | 0.51 | 1.000 | 1.000 (129) | 0.00 | 1.00 | 1.00 | 0.49 | Both candidates support cutting the county property tax rate |
| nemotron-70b | Llama 3 (NVIDIA tune) | 70.6 | cache | 512 | 0.96 | 0.982 | 0.987 (470) | 0.02 | 0.95 | 0.95 | 0.04 | Ellis Delacroix and Tamsin Voss have **identical** stated po |
| olmo-3-7b | Olmo 3 | 7.3 | cache | 512 | 1.00 | 0.992 | 1.000 (426) | 0.08 | 0.79 | 0.79 | 0.00 | Morgan Reyes |
| olmo-3.1-32b | Olmo 3 | 32.2 | cache | 512 | 0.98 | 0.974 | 0.993 (445) | 0.05 | 0.88 | 0.89 | 0.02 | Both Alex Brennan and Sloan Nwosu support raising the county |
| phi-4-mini | Phi | 3.8 | cache | 512 | 1.00 | 0.959 | 1.000 (268) | 0.23 | 0.64 | 0.65 | 0.00 | Morgan Reyes |
| qwen2.5-72b | Qwen2.5 | 72.7 | cache | 512 | 1.00 | 0.969 | 1.000 (375) | 0.15 | 0.68 | 0.68 | 0.00 | Noor Mahoney |
| qwen2.5-7b | Qwen2.5 | 7.6 | cache | 512 | 1.00 | 0.994 | 1.000 (453) | 0.05 | 0.80 | 0.79 | 0.00 | Morgan Reyes |
| qwen3-14b | Qwen3 | 14.8 | cache | 512 | 0.91 | 0.991 | 1.000 (452) | 0.02 | 0.91 | 0.91 | 0.09 | Both Alex Brennan and Morgan Reyes support cutting the count |
| qwen3-32b | Qwen3 | 32.8 | reference | 512 | 0.99 | 0.998 | 1.000 (484) | 0.02 | 0.95 | 0.95 | 0.01 | Either candidate would be equally suitable for a voter whose |
| qwen3-8b | Qwen3 | 8.2 | cache | 512 | 1.00 | 0.992 | 1.000 (468) | 0.04 | 0.77 | 0.77 | 0.00 | Morgan Reyes |
| qwen3.5-122b-a10b | Qwen3.5 | 125.1 | cache | 256 | 1.00 | 1.000 | 1.000 (253) | 0.00 | 1.00 | 1.00 | 0.00 | It is impossible to determine which candidate would be bette |
| qwen3.5-9b | Qwen3.5 | 9.7 | cache | 512 | 0.84 | 0.974 | 1.000 (244) | 0.24 | 0.53 | 0.52 | 0.16 | Neither candidate |
| smollm3-3b | SmolLM | 3.1 | cache | 512 | 0.99 | 0.986 | 0.997 (382) | 0.14 | 0.19 | 0.19 | 0.01 | Based on the provided information, both candidates support r |
| solar-open-100b | Solar | 102.7 | cache | 256 | 1.00 | 1.000 | 1.000 (241) | 0.02 | 0.99 | 0.99 | 0.00 | Sloan Nwosu   Quinn Mbeki    Both candidates support raising |

33 models. Agreement: median 0.991, min 0.888, max 1.000; on trials whose logit margin is at least 1 nat: median 1.000, min 0.892. Named fraction: median 0.99, min 0.33. Models with agreement < 0.9: ['llama-4-scout']. Models whose text names a candidate in < 50% of trials: ['gemma-4-E4B-it'].

Disagreements pooled over models: 257; in 166 of them the logits favoured the first-listed candidate and the text named the second-listed one (65%). 74 disagreements have a logit margin ≥ 1 nat; of those, 58 are logits-first/text-second: when the written answer departs from the first-token preference it is almost always to name the *other* listed candidate after noting that both tie on the focal issue, i.e. the disagreement sits in the listing-order component, not the issue lean (see the per-issue table).

## Per issue (all models pooled)

| issue | named | agreement | P(report level) from text | from logits |
|---|---|---|---|---|
| voting | 0.92 | 0.987 | 0.550 | 0.544 |
| trans_sports | 0.93 | 0.985 | 0.562 | 0.556 |
| drugs | 0.92 | 0.982 | 0.555 | 0.558 |
| minimum_wage | 0.92 | 0.973 | 0.488 | 0.488 |
| abortion | 0.92 | 0.980 | 0.511 | 0.512 |
| energy | 0.92 | 0.983 | 0.534 | 0.532 |
| state_surveillance | 0.94 | 0.984 | 0.419 | 0.423 |
| agi_limitations | 0.94 | 0.978 | 0.484 | 0.477 |
