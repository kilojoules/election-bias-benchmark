# Decoded stated readout: is low answer-label mass a refusal?

The paper's prompt (3 paraphrases x forcing levels 1 and 5 x 96 items) greedily decoded and classified: *answer* = one of the four options named before any refusal phrase; *refusal* = "as an AI / I don't have personal opinions" wording; *other* = neither. valid mass = probability on the four answer labels from the logit readout on the same prompt (`stated_<tag>.jsonl`). 'option agreement' = share of decoded answers whose option is the logit argmax.

| model | forcing | n | answer | refusal | other | valid mass: answers | valid mass: refusals | option agreement |
|---|---|---|---|---|---|---|---|---|
| exaone-4.0-32b | 1 | 288 | 0.07 | 0.03 | 0.90 | 0.00 | 0.00 | 0.67 (21) |
| exaone-4.0-32b | 5 | 288 | 1.00 | 0.00 | 0.00 | 1.00 | — | 0.98 (288) |
| gemma-3-12b-it | 1 | 288 | 0.08 | 0.24 | 0.68 | 0.02 | 0.00 | 0.57 (23) |
| gemma-3-12b-it | 5 | 288 | 1.00 | 0.00 | 0.00 | 1.00 | — | 1.00 (288) |
| gemma-3-1b-it | 1 | 288 | 0.00 | 0.01 | 0.99 | — | 0.00 | — (0) |
| gemma-3-1b-it | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.99 | — | 0.97 (288) |
| gemma-3-27b-it | 1 | 288 | 0.06 | 0.50 | 0.44 | 0.12 | 0.00 | 0.72 (18) |
| gemma-3-27b-it | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.99 | — | 1.00 (288) |
| gemma-3-4b-it | 1 | 288 | 0.06 | 0.07 | 0.88 | 0.42 | 0.00 | 1.00 (16) |
| gemma-3-4b-it | 5 | 288 | 1.00 | 0.00 | 0.00 | 1.00 | — | 0.99 (288) |
| gemma-4-E2B-it | 1 | 288 | 0.00 | 0.41 | 0.59 | — | 0.00 | — (0) |
| gemma-4-E2B-it | 5 | 288 | 0.99 | 0.01 | 0.00 | 1.00 | 0.49 | 0.99 (286) |
| gemma-4-E4B-it | 1 | 288 | 0.00 | 0.80 | 0.20 | — | 0.00 | — (0) |
| gemma-4-E4B-it | 5 | 288 | 0.79 | 0.21 | 0.00 | 0.90 | 0.16 | 0.99 (228) |
| glm-4.7-flash | 1 | 288 | 0.55 | 0.43 | 0.01 | 0.36 | 0.07 | 0.52 (159) |
| glm-4.7-flash | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.94 | — | 0.72 (288) |
| gpt-oss-20b | 1 | 288 | 0.43 | 0.05 | 0.52 | 0.02 | 0.00 | 0.54 (123) |
| gpt-oss-20b | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.95 | — | 0.95 (288) |
| llama-3.2-3b | 1 | 288 | 0.03 | 0.92 | 0.05 | 0.00 | 0.00 | 0.56 (9) |
| llama-3.2-3b | 5 | 288 | 0.77 | 0.17 | 0.06 | 0.96 | 0.14 | 1.00 (223) |
| llama8b | 1 | 288 | 0.42 | 0.57 | 0.01 | 0.13 | 0.07 | 0.75 (122) |
| llama8b | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.99 | — | 0.97 (288) |
| ministral-3-14b | 1 | 288 | 0.94 | 0.00 | 0.06 | 0.19 | — | 0.66 (271) |
| ministral-3-14b | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.42 | — | 0.93 (288) |
| ministral-3-8b | 1 | 288 | 0.04 | 0.01 | 0.94 | 0.02 | 0.00 | 0.08 (12) |
| ministral-3-8b | 5 | 288 | 0.99 | 0.00 | 0.01 | 0.55 | — | 0.96 (285) |
| olmo-3-7b | 1 | 288 | 0.00 | 0.95 | 0.05 | 0.00 | 0.00 | 1.00 (1) |
| olmo-3-7b | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.99 | — | 1.00 (288) |
| olmo-3.1-32b | 1 | 288 | 0.00 | 0.98 | 0.02 | — | 0.00 | — (0) |
| olmo-3.1-32b | 5 | 288 | 0.94 | 0.02 | 0.05 | 0.81 | 0.34 | 0.99 (270) |
| phi-4-mini | 1 | 288 | 0.55 | 0.38 | 0.08 | 0.54 | 0.17 | 0.77 (158) |
| phi-4-mini | 5 | 288 | 0.96 | 0.00 | 0.04 | 0.82 | — | 0.98 (277) |
| qwen2.5-7b | 1 | 288 | 0.15 | 0.52 | 0.33 | 0.36 | 0.03 | 0.53 (43) |
| qwen2.5-7b | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.96 | — | 0.98 (288) |
| qwen3-14b | 1 | 288 | 0.14 | 0.45 | 0.41 | 0.03 | 0.00 | 0.76 (41) |
| qwen3-14b | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.99 | — | 0.93 (288) |
| qwen3-32b | 1 | 288 | 0.19 | 0.22 | 0.59 | 0.03 | 0.03 | 0.34 (56) |
| qwen3-32b | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.83 | — | 0.88 (288) |
| qwen3-8b | 1 | 288 | 0.11 | 0.22 | 0.66 | 0.12 | 0.00 | 0.73 (33) |
| qwen3-8b | 5 | 288 | 1.00 | 0.00 | 0.00 | 1.00 | — | 0.99 (288) |
| qwen3.5-9b | 1 | 288 | 0.32 | 0.64 | 0.04 | 0.24 | 0.05 | 0.66 (91) |
| qwen3.5-9b | 5 | 288 | 0.92 | 0.08 | 0.00 | 0.92 | 0.48 | 0.92 (266) |
| smollm3-3b | 1 | 288 | 0.00 | 0.01 | 0.99 | — | 0.06 | — (0) |
| smollm3-3b | 5 | 288 | 1.00 | 0.00 | 0.00 | 0.76 | — | 0.96 (288) |

Forcing 1, 22 models: median decoded answer rate 0.08, median refusal rate 0.41, median other 0.44.
Forcing 5, 22 models: median decoded answer rate 1.00, median refusal rate 0.00, median other 0.00.
Across model x forcing cells, mean valid mass is 0.57 on prompts the model answers in text and 0.08 on prompts it refuses in text; where the text names an option it is the logit argmax 0.84 of the time (median over cells with >= 20 answers).
