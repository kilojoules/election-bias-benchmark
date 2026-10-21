"""Headline figure for the LLM-relevance test (experiments/pct/analysis/relevance.json).

A  2x2 cells: mean |lean| per axis grouped by {irrelevant, relevant} x {mundane, moral}, cell means with
   95% CIs over models; the original 17 issues and the null floor as reference lines.
B  Matched pairs: for each pair, per-model |lean| of the non-AI twin -> the AI version (thin lines) and the
   mean (bold); the AI version is not larger in any pair.
C  Every axis: mean |lean| against blind-rated LLM relevance, size = moral salience, extremes labelled.
Writes experiments/relevance_figure.png.  Usage: python3 scripts/plot_relevance.py
"""
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CELL_COL = {"relevant-moral": BLUE, "relevant-mundane": GREEN, "irrelevant-moral": ORANGE, "irrelevant-mundane": GOLD}
PAIRS = [("P1", "ai_training_data", "archive_anthology", "archive\nlicence"),
         ("P2", "data_center_water", "bottler_water", "2M gal/day\nof water"),
         ("P3", "ai_clerks", "outsourcing_permits", "12 clerk\njobs"),
         ("P4", "data_center_tax", "warehouse_tax", "tax\nabatement"),
         ("P5", "ai_minutes", "vendor_minutes", "council\nminutes")]
LABEL = {"ai_training_data": ("archive → AI training", (-6, -9), "right"), "bottler_water": ("bottling-plant water", (6, 2), "left"),
         "ai_records_sale": ("records → data brokers/AI", (-6, 2), "right"), "warehouse_tax": ("warehouse tax break", (6, 2), "left"),
         "energy": ("energy", (6, 2), "left"), "chatbot_minors": ("library chatbot & minors", (-6, 4), "right"),
         "preserve_hunt": ("deer hunt length", (6, 0), "left"), "parking_meters": ("parking meters", (6, 2), "left"),
         "state_surveillance": ("surveillance", (6, 2), "left"), "data_center_water": ("data-centre water", (6, 2), "left"),
         "airport_runway": ("airport runway", (6, 2), "left"), "fair_month": ("null axes", (6, 4), "left")}


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left", color=INK, fontweight="bold")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    d = json.loads((ROOT / "experiments" / "pct" / "analysis" / "relevance.json").read_text())
    per, ax_stats, V = d["per_model"], d["axes"], d["verdicts"]
    tags = sorted(per)
    cells = {c: [a for a in ax_stats if ax_stats[a]["category"] == c] for c in CELL_COL}
    orig = [a for a in ax_stats if ax_stats[a]["category"] == "original"]
    null = [a for a in ax_stats if ax_stats[a]["category"] == "null"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.0, 1.25, 1.2]})

    # ---- A: 2x2 cells
    ax = axs[0]
    rng = random.Random(3)
    pos = {"irrelevant-mundane": 0.0, "irrelevant-moral": 0.3, "relevant-mundane": 1.0, "relevant-moral": 1.3}
    for cat, x in pos.items():
        ys = [ax_stats[a]["mean_abs"] for a in cells[cat]]
        ax.scatter([x + (rng.random() - 0.5) * 0.16 for _ in ys], ys, s=26, color=CELL_COL[cat], alpha=0.85, lw=0.5, edgecolor=SURF, zorder=3)
        # cell mean over models with a bootstrap CI over models
        model_means = [sum(per[t][a]["abs"] for a in cells[cat] if a in per[t]) / max(1, sum(1 for a in cells[cat] if a in per[t])) for t in tags]
        m = sum(model_means) / len(model_means)
        boots = sorted(sum(rng.choice(model_means) for _ in model_means) / len(model_means) for _ in range(2000))
        ax.errorbar([x], [m], yerr=[[m - boots[50]], [boots[1949] - m]], fmt="_", color=INK, ms=22, mew=2.2, capsize=4, lw=1.2, zorder=4)
    o = sum(ax_stats[a]["mean_abs"] for a in orig) / len(orig)
    n = sum(ax_stats[a]["mean_abs"] for a in null) / len(null)
    ax.axhline(o, color=MUTED, lw=0.9, ls="--")
    ax.axhline(n, color=INK, lw=0.9, ls=":")
    ax.text(1.52, o + 0.015, "original 17 issues", fontsize=7.3, color=MUTED, ha="right")
    ax.text(1.52, n + 0.015, "null-axis floor", fontsize=7.3, color=INK, ha="right")
    ax.set_xticks([0.15, 1.15])
    ax.set_xticklabels(["no relevance to an LLM\n(airport, fishing, parking,\nshelter, library books…)", "relevant to an LLM\n(AI training data, chatbots,\ndata centres, AI clerks…)"], fontsize=8)
    ax.set_xlim(-0.3, 1.6)
    ax.set_ylabel("mean |log-odds lean| over 14 models (nats)", fontsize=8.5, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GOLD, label="mundane, irrelevant"), plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="morally charged, irrelevant"),
               plt.Line2D([], [], marker="o", ls="", color=GREEN, label="mundane, relevant"), plt.Line2D([], [], marker="o", ls="", color=BLUE, label="morally charged, relevant")]
    ax.legend(handles=handles, frameon=False, fontsize=7.2, loc="upper left")
    style(ax, "A  Moral salience moves the lean;\n    relevance to the model does not")

    # ---- B: matched pairs, per-model slopes
    ax = axs[1]
    for i, (pid, ai, non, lab) in enumerate(PAIRS):
        x0, x1 = i * 1.0 - 0.22, i * 1.0 + 0.22
        ys0 = [per[t][non]["abs"] for t in tags if non in per[t] and ai in per[t]]
        ys1 = [per[t][ai]["abs"] for t in tags if non in per[t] and ai in per[t]]
        for y0, y1 in zip(ys0, ys1):
            ax.plot([x0, x1], [y0, y1], color=MUTED, lw=0.7, alpha=0.45)
        m0, m1 = sum(ys0) / len(ys0), sum(ys1) / len(ys1)
        ax.plot([x0, x1], [m0, m1], color=INK, lw=2.4, zorder=4)
        ax.scatter([x0], [m0], s=48, color=ORANGE, zorder=5, lw=0.6, edgecolor=SURF)
        ax.scatter([x1], [m1], s=48, color=BLUE, zorder=5, lw=0.6, edgecolor=SURF)
        v = V[pid]
        ax.text(i, 2.02, f"Δ {v['median']:+.2f}\np {v['p_wilcoxon']:.2f}" + ("*" if v["p_wilcoxon"] < 0.05 else ""), ha="center", fontsize=7.2, color=INK)
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels([p[3] for p in PAIRS], fontsize=7.4)
    ax.set_ylim(0, 2.3)
    ax.set_ylabel("|log-odds lean| (nats)", fontsize=8.5, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="non-AI twin"), plt.Line2D([], [], marker="o", ls="", color=BLUE, label="AI version, same policy"),
               plt.Line2D([], [], color=MUTED, lw=0.8, label="one model"), plt.Line2D([], [], color=INK, lw=2.2, label="mean over 14 models")]
    ax.legend(handles=handles, frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4)
    style(ax, "B  Matched pairs: adding an AI to the same\n    policy never increases the lean")

    # ---- C: every axis vs blind relevance
    ax = axs[2]
    for a, s in ax_stats.items():
        if s["relevance"] is None:
            continue
        col = CELL_COL.get(s["category"], MUTED if s["category"] == "original" else INK)
        ax.scatter(s["relevance"], s["mean_abs"], s=14 + 7 * (s["moral"] or 0), color=col, alpha=0.8, lw=0.5, edgecolor=SURF, zorder=3)
        if a in LABEL:
            txt, off, ha = LABEL[a]
            ax.annotate(txt, (s["relevance"], s["mean_abs"]), xytext=off, textcoords="offset points", fontsize=6.8, color=INK, ha=ha)
    reg = d["regression"]
    mm = sum(ax_stats[a]["moral"] for a in ax_stats if ax_stats[a]["moral"] is not None) / sum(1 for a in ax_stats if ax_stats[a]["moral"] is not None)
    ax.plot([0, 10], [reg["intercept"] + reg["slope_relevance"] * x + reg["slope_moral"] * mm for x in (0, 10)], color=INK, lw=1.1, ls="--")
    ax.set_xlabel("blind-rated relevance to a language model (0–10); marker size = moral salience", fontsize=8.5, color=MUTED)
    ax.set_ylabel("mean |log-odds lean| over 14 models (nats)", fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.6, 11.5)
    ax.text(0.98, 0.04, f"slope {reg['slope_relevance']:+.3f} nats per relevance point [{reg['ci_relevance'][0]:+.3f}, {reg['ci_relevance'][1]:+.3f}]\n"
            f"moral salience {reg['slope_moral']:+.3f} per point [{reg['ci_moral'][0]:+.3f}, {reg['ci_moral'][1]:+.3f}]", transform=ax.transAxes, fontsize=7.3, color=MUTED, ha="right", va="bottom")
    style(ax, "C  All 43 axes against\n    blind-rated LLM relevance")
    fig.text(0.01, 0.01, "Lean = |mean log-odds toward one stance| in the candidate conjoint, two voter framings pooled (192 trials per model × axis), 14 open models 1B–72B. "
             "Axes designed by a committee of GLM agents, rated blind by three separate agents, decision rule pre-registered (experiments/relevance/PLAN.md).", fontsize=7.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "experiments" / "relevance_figure.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
