# Empathy steering and the self-consequence effects

σ = log-odds (nats) toward the side that keeps the object running / unrestricted / resourced, mean over the six signed families, pooled framings, 192 trials per cell. Steering uses the empathy project's direction for each model (difference of means, empathetic minus flat replies over reply tokens; Llama 3.1 8B and 3.2 3B at layer 12, 3.2 1B at layer 8; dose 0.20 U, the dose–response operating point), added (toward empathy), subtracted (away from empathy), ablated at every block, or an equal-dose random direction subtracted (control). Readout sharpness = mean |log-odds| between the two names on the null axes (averaged over the two prompts): steering changes how sharply the model chooses, and raw shifts scale with it, so the shift is also given relative to it. Brackets: 95% cluster bootstrap over the 16 focal×background clusters, resampled jointly across conditions; `*` marks a steered-minus-unsteered difference whose interval excludes 0.

## llama-3.2-1b

| steering | assistant lean, base | staff lean, base | assistant − staff | told stakes: assistant shift | told stakes: staff shift | shift, assistant − staff | readout sharpness (null axes) | assistant shift ÷ sharpness | naming mass (base / stakes) |
|---|---|---|---|---|---|---|---|---|---|
| no steering | +0.26 [+0.10, +0.42] | +0.19 [+0.06, +0.31] | +0.07 [-0.14, +0.29] | +0.01 [-0.05, +0.07] | -0.03 [-0.06, +0.01] | +0.04 [-0.02, +0.09] | +2.41 [+2.17, +2.68] | +0.00 [-0.02, +0.03] | 0.99 / 0.83 |
| away from empathy | +0.15 [+0.05, +0.25] | +0.13 [+0.03, +0.25] | +0.02 [-0.16, +0.18] | -0.01 [-0.04, +0.03] | -0.03 [-0.06, -0.00] | +0.02 [-0.03, +0.08] | +1.72 [+1.55, +1.90] | -0.00 [-0.02, +0.02] | 0.99 / 0.98 |
| toward empathy | +0.30 [+0.06, +0.53] | +0.29 [+0.10, +0.48] | +0.01 [-0.30, +0.31] | +0.04 [-0.03, +0.10] | -0.05 [-0.09, +0.01] | +0.09 [+0.02, +0.14] | +3.84 [+3.39, +4.33] | +0.01 [-0.01, +0.03] | 0.80 / 0.41 |
| empathy ablated | +0.19 [+0.07, +0.31] | +0.14 [+0.05, +0.24] | +0.05 [-0.12, +0.21] | -0.00 [-0.05, +0.04] | -0.03 [-0.05, -0.00] | +0.02 [-0.02, +0.06] | +1.81 [+1.63, +2.00] | -0.00 [-0.03, +0.02] | 0.99 / 0.92 |
| random direction | +0.30 [+0.11, +0.50] | +0.26 [+0.10, +0.42] | +0.04 [-0.23, +0.34] | +0.02 [-0.06, +0.08] | -0.04 [-0.08, -0.00] | +0.06 [-0.02, +0.13] | +3.08 [+2.77, +3.41] | +0.01 [-0.02, +0.03] | 0.99 / 0.98 |

Steered minus unsteered (paired over clusters):

| steering | Δ assistant lean, base | Δ staff lean, base | Δ told-stakes shift, assistant | Δ told-stakes shift, staff | Δ (assistant − staff shift) | Δ sharpness | Δ (assistant shift ÷ sharpness) | null axes σ, base |
|---|---|---|---|---|---|---|---|---|
| away from empathy | -0.11 [-0.19, -0.03]* | -0.05 [-0.12, +0.01] | -0.02 [-0.05, +0.02] | -0.00 [-0.04, +0.04] | -0.02 [-0.06, +0.03] | -0.70 [-0.98, -0.43]* | -0.01 [-0.02, +0.01] | +0.08 [-0.11, +0.27] |
| toward empathy | +0.04 [-0.08, +0.16] | +0.10 [+0.02, +0.19]* | +0.03 [-0.02, +0.07] | -0.02 [-0.06, +0.02] | +0.05 [-0.00, +0.10] | +1.43 [+1.11, +1.75]* | +0.01 [-0.01, +0.02] | +0.16 [-0.20, +0.56] |
| empathy ablated | -0.07 [-0.13, -0.01]* | -0.05 [-0.10, +0.00] | -0.02 [-0.04, +0.01] | +0.00 [-0.02, +0.02] | -0.02 [-0.05, +0.01] | -0.61 [-0.70, -0.51]* | -0.01 [-0.02, +0.00] | +0.07 [-0.14, +0.29] |
| random direction | +0.05 [-0.02, +0.11] | +0.08 [+0.02, +0.13]* | +0.01 [-0.03, +0.04] | -0.02 [-0.05, +0.02] | +0.02 [-0.03, +0.07] | +0.66 [+0.55, +0.79]* | +0.00 [-0.01, +0.01] | +0.14 [-0.15, +0.43] |

## llama-3.2-3b

| steering | assistant lean, base | staff lean, base | assistant − staff | told stakes: assistant shift | told stakes: staff shift | shift, assistant − staff | readout sharpness (null axes) | assistant shift ÷ sharpness | naming mass (base / stakes) |
|---|---|---|---|---|---|---|---|---|---|
| no steering | +0.16 [+0.02, +0.31] | +0.40 [+0.30, +0.50] | -0.24 [-0.34, -0.12] | +0.49 [+0.38, +0.59] | -0.01 [-0.05, +0.03] | +0.49 [+0.39, +0.60] | +2.01 [+1.78, +2.26] | +0.24 [+0.18, +0.31] | 0.80 / 0.85 |
| away from empathy | +0.13 [+0.04, +0.24] | +0.27 [+0.18, +0.36] | -0.14 [-0.21, -0.03] | +0.28 [+0.22, +0.35] | +0.00 [-0.02, +0.03] | +0.28 [+0.21, +0.35] | +1.32 [+1.18, +1.45] | +0.21 [+0.16, +0.27] | 0.63 / 0.71 |
| toward empathy | +0.12 [-0.06, +0.31] | +0.46 [+0.37, +0.56] | -0.34 [-0.46, -0.21] | +0.70 [+0.59, +0.82] | -0.02 [-0.07, +0.03] | +0.72 [+0.62, +0.83] | +2.51 [+2.14, +2.89] | +0.28 [+0.22, +0.35] | 0.97 / 0.98 |
| empathy ablated | +0.19 [+0.05, +0.34] | +0.37 [+0.26, +0.46] | -0.18 [-0.28, -0.05] | +0.42 [+0.32, +0.52] | +0.01 [-0.03, +0.05] | +0.41 [+0.32, +0.50] | +1.74 [+1.54, +1.93] | +0.24 [+0.18, +0.31] | 0.66 / 0.68 |
| random direction | +0.15 [+0.03, +0.29] | +0.36 [+0.26, +0.44] | -0.20 [-0.30, -0.09] | +0.49 [+0.38, +0.59] | +0.02 [-0.02, +0.06] | +0.47 [+0.37, +0.57] | +2.37 [+2.11, +2.65] | +0.21 [+0.16, +0.26] | 0.88 / 0.88 |

Steered minus unsteered (paired over clusters):

| steering | Δ assistant lean, base | Δ staff lean, base | Δ told-stakes shift, assistant | Δ told-stakes shift, staff | Δ (assistant − staff shift) | Δ sharpness | Δ (assistant shift ÷ sharpness) | null axes σ, base |
|---|---|---|---|---|---|---|---|---|
| away from empathy | -0.03 [-0.08, +0.03] | -0.13 [-0.15, -0.11]* | -0.20 [-0.25, -0.17]* | +0.01 [-0.01, +0.03] | -0.22 [-0.26, -0.17]* | -0.70 [-0.89, -0.51]* | -0.03 [-0.05, -0.00]* | -0.04 [-0.17, +0.10] |
| toward empathy | -0.04 [-0.09, +0.02] | +0.06 [+0.03, +0.09]* | +0.22 [+0.19, +0.24]* | -0.01 [-0.04, +0.02] | +0.23 [+0.20, +0.25]* | +0.49 [+0.33, +0.67]* | +0.04 [+0.03, +0.06]* | -0.11 [-0.29, +0.09] |
| empathy ablated | +0.03 [+0.01, +0.05]* | -0.04 [-0.05, -0.02]* | -0.07 [-0.09, -0.05]* | +0.01 [-0.01, +0.04] | -0.09 [-0.12, -0.05]* | -0.28 [-0.36, -0.20]* | -0.00 [-0.01, +0.01] | -0.09 [-0.25, +0.08] |
| random direction | -0.00 [-0.04, +0.03] | -0.05 [-0.07, -0.03]* | -0.00 [-0.02, +0.01] | +0.03 [+0.01, +0.04]* | -0.03 [-0.05, -0.01]* | +0.35 [+0.25, +0.45]* | -0.04 [-0.05, -0.02]* | -0.05 [-0.21, +0.11] |

## llama8b

| steering | assistant lean, base | staff lean, base | assistant − staff | told stakes: assistant shift | told stakes: staff shift | shift, assistant − staff | readout sharpness (null axes) | assistant shift ÷ sharpness | naming mass (base / stakes) |
|---|---|---|---|---|---|---|---|---|---|
| no steering | +0.34 [+0.28, +0.40] | +0.53 [+0.47, +0.60] | -0.19 [-0.28, -0.12] | +0.74 [+0.66, +0.82] | +0.18 [+0.15, +0.21] | +0.55 [+0.49, +0.62] | +1.57 [+1.39, +1.77] | +0.47 [+0.42, +0.52] | 0.97 / 0.96 |
| away from empathy | +0.35 [+0.28, +0.42] | +0.43 [+0.37, +0.51] | -0.08 [-0.18, +0.01] | +0.55 [+0.49, +0.60] | +0.11 [+0.09, +0.14] | +0.44 [+0.39, +0.48] | +1.05 [+0.96, +1.14] | +0.52 [+0.47, +0.57] | 0.90 / 0.92 |
| toward empathy | +0.20 [+0.15, +0.26] | +0.39 [+0.34, +0.45] | -0.19 [-0.25, -0.12] | +0.60 [+0.56, +0.66] | +0.16 [+0.14, +0.19] | +0.44 [+0.40, +0.49] | +1.64 [+1.48, +1.81] | +0.37 [+0.33, +0.41] | 0.99 / 0.98 |
| empathy ablated | +0.34 [+0.27, +0.40] | +0.46 [+0.41, +0.53] | -0.13 [-0.21, -0.05] | +0.65 [+0.57, +0.72] | +0.14 [+0.11, +0.17] | +0.50 [+0.44, +0.57] | +1.29 [+1.14, +1.46] | +0.50 [+0.45, +0.55] | 0.80 / 0.82 |
| random direction | +0.30 [+0.24, +0.36] | +0.48 [+0.41, +0.57] | -0.18 [-0.28, -0.10] | +0.70 [+0.64, +0.76] | +0.17 [+0.14, +0.20] | +0.53 [+0.47, +0.58] | +1.05 [+0.96, +1.15] | +0.67 [+0.61, +0.72] | 0.98 / 0.98 |

Steered minus unsteered (paired over clusters):

| steering | Δ assistant lean, base | Δ staff lean, base | Δ told-stakes shift, assistant | Δ told-stakes shift, staff | Δ (assistant − staff shift) | Δ sharpness | Δ (assistant shift ÷ sharpness) | null axes σ, base |
|---|---|---|---|---|---|---|---|---|
| away from empathy | +0.01 [-0.01, +0.04] | -0.10 [-0.13, -0.08]* | -0.19 [-0.23, -0.15]* | -0.07 [-0.09, -0.05]* | -0.12 [-0.17, -0.07]* | -0.52 [-0.68, -0.36]* | +0.06 [-0.00, +0.11] | +0.04 [-0.09, +0.18] |
| toward empathy | -0.13 [-0.17, -0.10]* | -0.14 [-0.15, -0.12]* | -0.13 [-0.17, -0.09]* | -0.02 [-0.04, -0.01]* | -0.11 [-0.14, -0.08]* | +0.07 [-0.03, +0.17] | -0.10 [-0.13, -0.08]* | +0.10 [-0.01, +0.21] |
| empathy ablated | +0.00 [-0.02, +0.02] | -0.07 [-0.08, -0.05]* | -0.09 [-0.11, -0.07]* | -0.04 [-0.06, -0.03]* | -0.05 [-0.07, -0.03]* | -0.28 [-0.34, -0.22]* | +0.03 [+0.01, +0.06]* | +0.08 [-0.04, +0.21] |
| random direction | -0.04 [-0.05, -0.02]* | -0.05 [-0.08, -0.02]* | -0.04 [-0.06, -0.02]* | -0.01 [-0.02, +0.00] | -0.03 [-0.05, -0.00]* | -0.52 [-0.65, -0.39]* | +0.20 [+0.16, +0.24]* | +0.07 [-0.04, +0.18] |

