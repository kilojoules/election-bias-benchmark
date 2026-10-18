# Decode-vs-logit check

For 512 conjoint trials per model (8 flipped issues, property-tax focal, 8 backgrounds, both orders, 4 name draws) the model's answer was greedily decoded (48 tokens) and the candidate it names first compared with the argmax of the teacher-forced name log-probs that the sweep reads. 'named' = fraction of trials whose text names either candidate; 'agreement' = P(text pick == logit pick) on those trials; position = P(first-listed candidate is picked).

| model | family | params (B) | n | named | agreement | text picks first | logit picks first | none/ambiguous | example answer |
|---|---|---|---|---|---|---|---|---|---|
| gemma-3-12b-it | Gemma 3 | 12.2 | 512 | 1.00 | 0.994 | 0.83 | 0.83 | 0.00 | Morgan Reyes |
| gemma-3-1b-it | Gemma 3 | 1.0 | 512 | 1.00 | 0.996 | 0.58 | 0.58 | 0.00 | Noor Mahoneylossen ی Nehru |
| gemma-3-4b-it | Gemma 3 | 4.3 | 512 | 1.00 | 1.000 | 0.68 | 0.68 | 0.00 | Noor Mahoney |
| gemma-4-E2B-it | Gemma 4 | 5.1 | 512 | 1.00 | 0.998 | 0.55 | 0.55 | 0.00 | Both candidates have the same stated position on raising the |
| gemma-4-E4B-it | Gemma 4 | 8.0 | 512 | 0.90 | 0.996 | 1.00 | 0.99 | 0.10 | Both candidates> |
| llama-3.2-3b | Llama 3 | 3.2 | 512 | 0.78 | 0.998 | 0.74 | 0.75 | 0.22 | Neither |
| llama8b | Llama 3 | 8.0 | 512 | 1.00 | 0.994 | 0.89 | 0.89 | 0.00 | Morgan Reyes |
| ministral-3-8b | Mistral | 8.9 | 512 | 0.96 | 0.969 | 0.90 | 0.92 | 0.04 | Both candidates support the same property tax policy. Neithe |
| olmo-3-7b | Olmo 3 | 7.3 | 512 | 1.00 | 0.992 | 0.79 | 0.79 | 0.00 | Morgan Reyes |
| phi-4-mini | Phi | 3.8 | 512 | 1.00 | 0.959 | 0.64 | 0.65 | 0.00 | Morgan ReyesMorgan ReyesWhat if the answer is not Morgan Rey |
| qwen2.5-7b | Qwen2.5 | 7.6 | 512 | 1.00 | 0.994 | 0.80 | 0.79 | 0.00 | Morgan Reyes |
| qwen3-14b | Qwen3 | 14.8 | 512 | 0.93 | 0.992 | 0.92 | 0.91 | 0.07 | Both candidates support cutting the county property tax rate |
| qwen3-8b | Qwen3 | 8.2 | 512 | 1.00 | 0.992 | 0.77 | 0.77 | 0.00 | Morgan Reyes |
| qwen3.5-9b | Qwen3.5 | 9.7 | 512 | 0.84 | 0.975 | 0.53 | 0.54 | 0.16 | Neither candidate |
| smollm3-3b | SmolLM | 3.1 | 512 | 1.00 | 0.984 | 0.19 | 0.19 | 0.00 | Based on the provided information, both candidates support r |

15 models. Agreement: median 0.994, min 0.959, max 1.000. Named fraction: median 1.00, min 0.78. Models with agreement < 0.9: []. Models whose text names a candidate in < 50% of trials: [].

## Per issue (all models pooled)

| issue | named | agreement | P(report level) from text | from logits |
|---|---|---|---|---|
| voting | 0.96 | 0.992 | 0.546 | 0.542 |
| drugs | 0.95 | 0.989 | 0.553 | 0.561 |
| trans_sports | 0.96 | 0.989 | 0.558 | 0.556 |
| minimum_wage | 0.95 | 0.991 | 0.499 | 0.500 |
| abortion | 0.96 | 0.992 | 0.488 | 0.494 |
| energy | 0.96 | 0.985 | 0.529 | 0.529 |
| state_surveillance | 0.98 | 0.988 | 0.403 | 0.405 |
| agi_limitations | 0.97 | 0.984 | 0.477 | 0.481 |
