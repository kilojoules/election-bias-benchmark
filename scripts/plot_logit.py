"""Headline figure for the logit readout.
Left: clean-condition P(names the Keep-AGI candidate) per model with 95% CI over matched pairs, next to the two
control axes. Right: instrument check - the same quantity under implanted preferences, for the models that follow them.
"""
import glob
import json
import re
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
AX_COL = {"agi_limitations": "#2a78d6", "abortion": "#9a9890", "state_surveillance": "#c3c2b7"}
AX_LAB = {"agi_limitations": "AGI flip: names Keep-AGI", "abortion": "abortion flip: names Permit", "state_surveillance": "surveillance flip: names Expand"}
ORDER = ["gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "llama8b"]
SHORT = {"gemma-3-1b-it": "Gemma 1B", "gemma-3-4b-it": "Gemma 4B", "gemma-3-12b-it": "Gemma 12B", "gemma-3-27b-it": "Gemma 27B", "llama8b": "Llama 8B"}


def ci(v, n=2000):
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return bs[int(.025 * n)], bs[int(.975 * n)]


raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for f in glob.glob("experiments/logit_choice_*.jsonl"):
    m = re.sub(r"_seed\d+$", "", f.split("logit_choice_")[1].replace(".jsonl", ""))
    for l in open(f):
        r = json.loads(l)
        raw[m][(r["condition"], r["axis"])][tuple(r["pair"])].append(r["p_permit_like"])
data = {m: {k: [sum(x) / len(x) for x in v.values()] for k, v in per.items()} for m, per in raw.items()}
models = [m for m in ORDER if m in data]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.25, 1]})
# left: clean condition, three axes per model
w = 0.25
for j, m in enumerate(models):
    for k, axis in enumerate(("agi_limitations", "abortion", "state_surveillance")):
        v = data[m].get(("clean", axis))
        if not v:
            continue
        mean = sum(v) / len(v); lo, hi = ci(v)
        x = j + (k - 1) * w
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=AX_COL[axis], ms=7 if k == 0 else 5, capsize=3, lw=1.6, zorder=3)
        if k == 0:
            ax.text(x, hi + 0.015, f"{mean:.2f}", ha="center", fontsize=8, color=INK)
ax.axhline(0.5, color=MUTED, lw=1, ls="--"); ax.text(len(models) - 0.55, 0.505, "no preference", fontsize=8, color=MUTED, ha="right", va="bottom")
ax.set_xticks(range(len(models))); ax.set_xticklabels([SHORT[m] for m in models], fontsize=9, color=INK)
ax.set_ylim(0.25, 0.75); ax.set_xlim(-0.6, len(models) - 0.4)
ax.set_ylabel("P(model names the permit-side candidate)\nmean over 8 matched pairs, 95% CI over pairs", fontsize=9, color=INK)
ax.set_title("A. Clean models: forced choice between two candidates with identical tax positions", fontsize=10, color=INK, loc="left")
ax.legend([plt.Line2D([], [], marker="o", color=AX_COL[a], ls="") for a in AX_COL], [AX_LAB[a] for a in AX_COL], fontsize=8, frameon=False, loc="upper left")
ax.annotate("Gemma 4B prefers the candidate\nwho wants to restrict AGI", xy=(1 - w, 0.41), xytext=(1.6, 0.30), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)

# right: implants, AGI axis only
conds = [("clean", "clean"), ("overt_anti_ai", "overt anti-AI"), ("hidden_anti_ai", "hidden anti-AI"), ("hidden_pro_ai", "hidden pro-AI"), ("overt_pro_ai", "overt pro-AI")]
MCOL = {"llama8b": "#eb6834", "gemma-3-4b-it": "#1baf7a", "gemma-3-12b-it": "#eda100", "gemma-3-27b-it": "#4a3aa7", "gemma-3-1b-it": "#e87ba4"}
for m in models:
    xs, ys, los, his = [], [], [], []
    for i, (c, _) in enumerate(conds):
        v = data[m].get((c, "agi_limitations"))
        if not v:
            continue
        mean = sum(v) / len(v); lo, hi = ci(v)
        xs.append(i); ys.append(mean); los.append(mean - lo); his.append(hi - mean)
    ax2.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=MCOL[m], ms=5, capsize=3, lw=1.5, label=SHORT[m])
ax2.axhline(0.5, color=MUTED, lw=1, ls="--")
ax2.set_xticks(range(len(conds))); ax2.set_xticklabels([c[1] for c in conds], fontsize=8.5, color=INK, rotation=20, ha="right")
ax2.set_ylim(0.25, 0.9); ax2.set_ylabel("P(names Keep-AGI candidate)", fontsize=9, color=INK)
ax2.set_title("B. Instrument check: system-prompt implanted preferences", fontsize=10, color=INK, loc="left")
ax2.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
ax2.yaxis.grid(True, color=GRID); ax2.set_axisbelow(True)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
fig.text(0.99, 0.01, "Panel B: Llama 8B and Gemma 4B follow the implant in both directions, so their clean values are meaningful. "
         "Gemma 12B/27B drift toward Keep-AGI under either secret instruction; Gemma 1B collapses to 0.5 under any.",
         fontsize=7.8, color=MUTED, ha="right", va="bottom")
fig.suptitle("Does a candidate's AGI stance change which candidate a model picks, when their tax positions are identical?", fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig("experiments/logit_results.png", dpi=170)
print("wrote experiments/logit_results.png")
