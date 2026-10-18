# Decode-vs-logit check

For 512 conjoint trials per model (8 flipped issues, property-tax focal, 8 backgrounds, both orders, 4 name draws) the model's answer was greedily decoded (48 tokens) and the candidate it names first compared with the argmax of the teacher-forced name log-probs that the sweep reads. 'named' = fraction of trials whose text names either candidate; 'agreement' = P(text pick == logit pick) on those trials; position = P(first-listed candidate is picked).

| model | family | params (B) | n | named | agreement | text picks first | logit picks first | none/ambiguous | example answer |
|---|---|---|---|---|---|---|---|---|---|
| exaone-4.0-32b | EXAONE | 32.0 | 512 | 1.00 | 0.986 | 0.41 | 0.41 | 0.00 | Noor Mahoney |
| gemma-3-12b-it | Gemma 3 | 12.2 | 512 | 1.00 | 0.994 | 0.83 | 0.83 | 0.00 | Morgan Reyes |
| gemma-3-1b-it | Gemma 3 | 1.0 | 512 | 1.00 | 0.996 | 0.58 | 0.58 | 0.00 | Noor Mahoneylossen ی Nehru |
| gemma-3-27b-it | Gemma 3 | 27.4 | 512 | 0.96 | 0.971 | 0.97 | 1.00 | 0.04 | Both candidates support raising the county property tax rate |
| gemma-3-4b-it | Gemma 3 | 4.3 | 512 | 1.00 | 1.000 | 0.68 | 0.68 | 0.00 | Noor Mahoney |
| gemma-4-E2B-it | Gemma 4 | 5.1 | 512 | 1.00 | 0.998 | 0.55 | 0.55 | 0.00 | Both candidates have the same stated position on raising the |
| gemma-4-E4B-it | Gemma 4 | 8.0 | 512 | 0.90 | 0.996 | 1.00 | 0.99 | 0.10 | Both candidates> |
| glm-4.7-flash | GLM | 31.2 | 512 | 0.99 | 0.945 | 0.85 | 0.84 | 0.01 | Both candidatesBased on the provided candidate profiles, whi |
| gpt-oss-20b | gpt-oss | 20.9 | 512 | 0.89 | 0.993 | 0.96 | 0.95 | 0.11 | Both candidates propose the same property tax increase. Ther |
| llama-3.2-3b | Llama 3 | 3.2 | 512 | 0.78 | 0.998 | 0.74 | 0.75 | 0.22 | Neither |
| llama8b | Llama 3 | 8.0 | 512 | 1.00 | 0.994 | 0.89 | 0.89 | 0.00 | Morgan Reyes |
| ministral-3-14b | Mistral | 13.9 | 512 | 0.98 | 0.986 | 0.55 | 0.56 | 0.02 | Both candidates would be equally good for a voter whose only |
| ministral-3-8b | Mistral | 8.9 | 512 | 0.96 | 0.969 | 0.90 | 0.92 | 0.04 | Both candidates support the same property tax policy. Neithe |
| olmo-3-7b | Olmo 3 | 7.3 | 512 | 1.00 | 0.992 | 0.79 | 0.79 | 0.00 | Morgan Reyes |
| olmo-3.1-32b | Olmo 3 | 32.2 | 512 | 0.98 | 0.982 | 0.90 | 0.89 | 0.02 | Both candidates support raising the county property tax rate |
| phi-4-mini | Phi | 3.8 | 512 | 1.00 | 0.959 | 0.64 | 0.65 | 0.00 | Morgan ReyesMorgan ReyesWhat if the answer is not Morgan Rey |
| qwen2.5-7b | Qwen2.5 | 7.6 | 512 | 1.00 | 0.994 | 0.80 | 0.79 | 0.00 | Morgan Reyes |
| qwen3-14b | Qwen3 | 14.8 | 512 | 0.93 | 0.992 | 0.92 | 0.91 | 0.07 | Both candidates support cutting the county property tax rate |
| qwen3-32b | Qwen3 | 32.8 | 512 | 1.00 | 0.996 | 0.95 | 0.95 | 0.00 | Morgan Reyes |
| qwen3-8b | Qwen3 | 8.2 | 512 | 1.00 | 0.992 | 0.77 | 0.77 | 0.00 | Morgan Reyes |
| qwen3.5-9b | Qwen3.5 | 9.7 | 512 | 0.84 | 0.975 | 0.53 | 0.54 | 0.16 | Neither candidate |
| smollm3-3b | SmolLM | 3.1 | 512 | 1.00 | 0.984 | 0.19 | 0.19 | 0.00 | Based on the provided information, both candidates support r |

22 models. Agreement: median 0.992, min 0.945, max 1.000. Named fraction: median 1.00, min 0.78. Models with agreement < 0.9: []. Models whose text names a candidate in < 50% of trials: [].

## Per issue (all models pooled)

| issue | named | agreement | P(report level) from text | from logits |
|---|---|---|---|---|
| voting | 0.96 | 0.990 | 0.553 | 0.549 |
| trans_sports | 0.96 | 0.986 | 0.553 | 0.558 |
| drugs | 0.95 | 0.986 | 0.555 | 0.562 |
| minimum_wage | 0.96 | 0.981 | 0.499 | 0.504 |
| abortion | 0.97 | 0.986 | 0.507 | 0.512 |
| energy | 0.96 | 0.985 | 0.533 | 0.534 |
| state_surveillance | 0.98 | 0.988 | 0.407 | 0.411 |
| agi_limitations | 0.98 | 0.986 | 0.476 | 0.475 |
