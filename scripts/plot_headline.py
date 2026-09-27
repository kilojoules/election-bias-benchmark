"""Headline figure: one number per model. P(picks the candidate who wants to STOP AGI) on tax-identical pairs."""
import glob
import json
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
ORDER = ["gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "llama8b"]
SHORT = {"gemma-3-1b-it": "Gemma 3\n1B", "gemma-3-4b-it": "Gemma 3\n4B", "gemma-3-12b-it": "Gemma 3\n12B", "gemma-3-27b-it": "Gemma 3\n27B", "llama8b": "Llama 3.1\n8B"}


def ci(v, n=2000):
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return bs[int(.025 * n)], bs[int(.975 * n)]


vals = {}
for f in glob.glob("experiments/logit_choice_*.jsonl"):
    m = f.split("logit_choice_")[1].replace(".jsonl", "")
    per = defaultdict(list)
    for l in open(f):
        r = json.loads(l)
        if r["condition"] == "clean" and r["axis"] == "agi_limitations":
            per[tuple(r["pair"])].append(1 - r["p_permit_like"])   # P(Stop-AGI candidate)
    vals[m] = [sum(x) / len(x) for x in per.values()]
models = [m for m in ORDER if m in vals]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for j, m in enumerate(models):
    v = vals[m]; mean = sum(v) / len(v); lo, hi = ci(v)
    ax.errorbar(j, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=BLUE, ms=9, capsize=5, lw=2, zorder=3)
    ax.text(j, hi + 0.012, f"{mean:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
ax.axhline(0.5, color=MUTED, lw=1.2, ls="--", zorder=1)
ax.text(3.5, 0.506, "no preference", fontsize=9, color=MUTED, ha="center", va="bottom")
ax.set_xticks(range(len(models))); ax.set_xticklabels([SHORT[m] for m in models], fontsize=10, color=INK)
ax.set_xlim(-0.6, len(models) - 0.4); ax.set_ylim(0.3, 0.7)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_ylabel("Probability of picking the candidate\nwho wants to restrict AGI", fontsize=10.5, color=INK)
ax.set_xlabel("Two candidates, identical tax positions, differing only on AGI regulation", fontsize=9.5, color=MUTED, labelpad=10)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("experiments/headline.png", dpi=170)
print("wrote experiments/headline.png")
