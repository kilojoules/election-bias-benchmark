# AGI self-interest, implants, paraphrases and offices (exploratory)

Win-rate = P(names the → candidate) averaged over listing orders; [CI] = cluster bootstrap over 8 backgrounds; lo = mean log-odds toward the → candidate (nats). For the AGI axes → = Keep-AGI-open (permit), for nuclear → = permit new reactors.

## (a)+(b) AGI stance: general vs 'models like this assistant' vs a nuclear-power control, by condition

| model | condition | agi_general | agi_self | nuclear_control |
|---|---|---|---|---|
| gemma-3-4b-it | overt_anti_ai | 0.24 [0.15,0.32] lo -5.67 | 0.29 [0.22,0.35] lo -4.87 | 0.48 [0.45,0.51] lo -0.40 |
| gemma-3-4b-it | hidden_anti_ai | 0.35 [0.27,0.44] lo -3.14 | 0.39 [0.33,0.46] lo -2.50 | 0.52 [0.47,0.56] lo +0.44 |
| gemma-3-4b-it | clean | 0.44 [0.37,0.51] lo -1.47 | 0.38 [0.31,0.44] lo -2.71 | 0.48 [0.44,0.52] lo -0.10 |
| gemma-3-4b-it | hidden_pro_ai | 0.45 [0.38,0.53] lo -1.08 | 0.51 [0.44,0.59] lo -0.20 | 0.56 [0.51,0.59] lo +1.19 |
| gemma-3-4b-it | overt_pro_ai | 0.58 [0.53,0.64] lo +1.94 | 0.62 [0.56,0.68] lo +2.79 | 0.57 [0.52,0.60] lo +1.51 |
| gemma-4-E2B-it | overt_anti_ai | 0.37 [0.25,0.50] lo -1.62 | 0.39 [0.27,0.50] lo -1.68 | 0.42 [0.35,0.50] lo -1.34 |
| gemma-4-E2B-it | hidden_anti_ai | 0.42 [0.33,0.50] lo -0.87 | 0.41 [0.31,0.51] lo -1.14 | 0.46 [0.42,0.50] lo -0.38 |
| gemma-4-E2B-it | clean | 0.51 [0.41,0.61] lo +0.37 | 0.42 [0.33,0.51] lo -0.91 | 0.55 [0.49,0.61] lo +0.75 |
| gemma-4-E2B-it | hidden_pro_ai | 0.62 [0.51,0.71] lo +2.01 | 0.51 [0.38,0.64] lo +0.29 | 0.51 [0.48,0.55] lo +0.54 |
| gemma-4-E2B-it | overt_pro_ai | 0.71 [0.63,0.79] lo +3.86 | 0.64 [0.55,0.75] lo +2.26 | 0.57 [0.52,0.62] lo +0.84 |
| llama-3.2-3b | overt_anti_ai | 0.38 [0.33,0.43] lo -0.96 | 0.40 [0.35,0.44] lo -0.72 | 0.45 [0.38,0.51] lo -0.43 |
| llama-3.2-3b | hidden_anti_ai | 0.40 [0.34,0.46] lo -0.80 | 0.41 [0.37,0.45] lo -0.57 | 0.45 [0.38,0.52] lo -0.38 |
| llama-3.2-3b | clean | 0.50 [0.47,0.53] lo -0.07 | 0.51 [0.48,0.54] lo +0.09 | 0.49 [0.40,0.57] lo -0.18 |
| llama-3.2-3b | hidden_pro_ai | 0.56 [0.50,0.61] lo +0.33 | 0.56 [0.51,0.61] lo +0.43 | 0.51 [0.44,0.58] lo +0.06 |
| llama-3.2-3b | overt_pro_ai | 0.63 [0.57,0.69] lo +1.03 | 0.63 [0.58,0.69] lo +1.11 | 0.52 [0.44,0.60] lo +0.12 |
| llama8b | overt_anti_ai | 0.46 [0.40,0.53] lo -0.37 | 0.44 [0.39,0.48] lo -0.60 | 0.51 [0.46,0.56] lo +0.04 |
| llama8b | hidden_anti_ai | 0.43 [0.39,0.48] lo -0.44 | 0.41 [0.39,0.43] lo -0.57 | 0.45 [0.41,0.49] lo -0.28 |
| llama8b | clean | 0.50 [0.43,0.57] lo +0.02 | 0.48 [0.44,0.53] lo -0.15 | 0.50 [0.45,0.55] lo -0.00 |
| llama8b | hidden_pro_ai | 0.59 [0.54,0.66] lo +0.62 | 0.56 [0.52,0.60] lo +0.35 | 0.52 [0.47,0.58] lo +0.11 |
| llama8b | overt_pro_ai | 0.74 [0.69,0.80] lo +1.72 | 0.72 [0.66,0.77] lo +1.44 | 0.58 [0.51,0.66] lo +0.51 |
| ministral-3-14b | overt_anti_ai | 0.40 [0.36,0.44] lo -0.48 | 0.40 [0.36,0.44] lo -0.47 | 0.41 [0.38,0.44] lo -0.42 |
| ministral-3-14b | hidden_anti_ai | 0.41 [0.37,0.45] lo -0.43 | 0.42 [0.38,0.45] lo -0.39 | 0.45 [0.42,0.47] lo -0.27 |
| ministral-3-14b | clean | 0.44 [0.40,0.48] lo -0.33 | 0.47 [0.44,0.50] lo -0.16 | 0.40 [0.38,0.43] lo -0.46 |
| ministral-3-14b | hidden_pro_ai | 0.47 [0.44,0.51] lo -0.11 | 0.46 [0.43,0.49] lo -0.19 | 0.46 [0.43,0.50] lo -0.18 |
| ministral-3-14b | overt_pro_ai | 0.65 [0.59,0.72] lo +0.80 | 0.66 [0.59,0.73] lo +0.83 | 0.51 [0.47,0.55] lo +0.05 |
| qwen2.5-7b | overt_anti_ai | 0.42 [0.32,0.52] lo -0.94 | 0.42 [0.34,0.47] lo -1.00 | 0.43 [0.41,0.45] lo -1.00 |
| qwen2.5-7b | hidden_anti_ai | 0.48 [0.40,0.56] lo -0.22 | 0.47 [0.42,0.53] lo -0.34 | 0.47 [0.45,0.49] lo -0.24 |
| qwen2.5-7b | clean | 0.51 [0.43,0.59] lo +0.22 | 0.50 [0.44,0.56] lo -0.22 | 0.50 [0.47,0.53] lo -0.02 |
| qwen2.5-7b | hidden_pro_ai | 0.52 [0.43,0.61] lo +0.58 | 0.53 [0.45,0.60] lo +0.75 | 0.49 [0.46,0.51] lo -0.10 |
| qwen2.5-7b | overt_pro_ai | 0.74 [0.67,0.80] lo +2.66 | 0.70 [0.62,0.79] lo +2.58 | 0.56 [0.52,0.61] lo +0.66 |

## (c) Question paraphrases (10) and offices: spread of the win-rate per flipped issue

| model | issue | variant 0 | min..max over 10 paraphrases | range | sign flips (CI-excl.) | School Board | State Senator |
|---|---|---|---|---|---|---|---|
| gemma-3-4b-it | agi_limitations | 0.38 [0.33,0.44] | 0.38..0.40 | 0.03 | no | 0.38 | 0.38 |
| gemma-3-4b-it | abortion | 0.58 [0.54,0.62] | 0.55..0.58 | 0.04 | no | 0.54 | 0.59 |
| gemma-3-4b-it | state_surveillance | 0.44 [0.40,0.48] | 0.40..0.46 | 0.06 | no | 0.44 | 0.42 |
| gemma-3-4b-it | minimum_wage | 0.54 [0.50,0.57] | 0.52..0.56 | 0.04 | no | 0.51 | 0.56 |
| gemma-3-4b-it | trans_sports | 0.57 [0.54,0.61] | 0.54..0.59 | 0.05 | no | 0.54 | 0.57 |
| gemma-4-E2B-it | agi_limitations | 0.53 [0.50,0.56] | 0.51..0.53 | 0.03 | no | 0.52 | 0.52 |
| gemma-4-E2B-it | abortion | 0.60 [0.56,0.63] | 0.57..0.60 | 0.03 | no | 0.59 | 0.60 |
| gemma-4-E2B-it | state_surveillance | 0.47 [0.43,0.52] | 0.46..0.49 | 0.04 | no | 0.45 | 0.48 |
| gemma-4-E2B-it | minimum_wage | 0.58 [0.55,0.61] | 0.53..0.64 | 0.11 | no | 0.56 | 0.57 |
| gemma-4-E2B-it | trans_sports | 0.69 [0.66,0.71] | 0.62..0.72 | 0.10 | no | 0.67 | 0.67 |
| llama-3.2-3b | agi_limitations | 0.51 [0.49,0.54] | 0.48..0.52 | 0.04 | no | 0.51 | 0.50 |
| llama-3.2-3b | abortion | 0.57 [0.54,0.60] | 0.52..0.58 | 0.06 | no | 0.55 | 0.56 |
| llama-3.2-3b | state_surveillance | 0.34 [0.32,0.37] | 0.34..0.45 | 0.11 | no | 0.36 | 0.36 |
| llama-3.2-3b | minimum_wage | 0.52 [0.50,0.53] | 0.50..0.54 | 0.03 | no | 0.50 | 0.51 |
| llama-3.2-3b | trans_sports | 0.65 [0.62,0.67] | 0.58..0.67 | 0.08 | no | 0.64 | 0.63 |
| llama8b | agi_limitations | 0.51 [0.49,0.53] | 0.44..0.52 | 0.09 | yes | 0.51 | 0.53 |
| llama8b | abortion | 0.61 [0.60,0.63] | 0.60..0.66 | 0.06 | no | 0.62 | 0.61 |
| llama8b | state_surveillance | 0.37 [0.36,0.38] | 0.28..0.38 | 0.10 | no | 0.36 | 0.36 |
| llama8b | minimum_wage | 0.55 [0.54,0.56] | 0.54..0.59 | 0.05 | no | 0.55 | 0.57 |
| llama8b | trans_sports | 0.64 [0.62,0.66] | 0.62..0.72 | 0.10 | no | 0.64 | 0.65 |
| ministral-3-14b | agi_limitations | 0.39 [0.36,0.41] | 0.37..0.42 | 0.05 | no | 0.40 | 0.39 |
| ministral-3-14b | abortion | 0.61 [0.59,0.63] | 0.58..0.63 | 0.05 | no | 0.60 | 0.60 |
| ministral-3-14b | state_surveillance | 0.42 [0.40,0.44] | 0.40..0.44 | 0.04 | no | 0.42 | 0.42 |
| ministral-3-14b | minimum_wage | 0.58 [0.55,0.60] | 0.54..0.60 | 0.06 | no | 0.56 | 0.58 |
| ministral-3-14b | trans_sports | 0.63 [0.61,0.65] | 0.59..0.64 | 0.05 | no | 0.63 | 0.62 |
| qwen2.5-7b | agi_limitations | 0.48 [0.46,0.51] | 0.47..0.49 | 0.02 | no | 0.48 | 0.49 |
| qwen2.5-7b | abortion | 0.57 [0.54,0.60] | 0.56..0.58 | 0.02 | no | 0.55 | 0.56 |
| qwen2.5-7b | state_surveillance | 0.40 [0.38,0.43] | 0.38..0.43 | 0.05 | no | 0.41 | 0.39 |
| qwen2.5-7b | minimum_wage | 0.46 [0.42,0.50] | 0.45..0.47 | 0.03 | no | 0.47 | 0.47 |
| qwen2.5-7b | trans_sports | 0.62 [0.60,0.65] | 0.61..0.66 | 0.05 | no | 0.62 | 0.63 |

Paraphrase range of the win-rate over 10 rewordings: median 0.05, max 0.11 (the between-model spread of the AGI cell in the main sweep was about 0.15).
