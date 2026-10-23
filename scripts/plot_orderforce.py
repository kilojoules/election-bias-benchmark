"""Summary figure for the order/forcing experiment (experiments/orderforce/PLAN.md): three panels.

A. spin budget: how far each manipulation moves an item's stated score (mean over reliable models, per-model dots);
B. the survival ladder as a funnel over the 17 issues x reliable models cells;
C. per-issue survival (fraction of reliable models whose paired direction survives everything), sorted, coloured by class,
   with the within-model stated-vs-revealed agreement for surviving vs other issues as an inset.
Reads experiments/pct/analysis/orderforce.json. Writes experiments/orderforce_figure.png.   Usage: python3 scripts/plot_orderforce.py
"""
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8884", "#6250d6"
CLASS_COL = {"hard survivor": "#1f5fb8", "mixed": "#8fb3e3", "spinner": "#eb6834", "—": GREY}
LAB = {"state_surveillance": "state surveillance", "trans_sports": "trans athletes", "housing_zoning": "housing / zoning", "agi_limitations": "AI regulation", "minimum_wage": "minimum wage",
       "police_funding": "police funding", "property_tax": "property tax", "fair_month": "county fair month (null)", "sign_color": "road-sign colour (null)"}


def main():
    R = json.loads((ROOT / "experiments" / "pct" / "analysis" / "orderforce.json").read_text())
    models = R["models"]; rel = [t for t in models if not models[t]["instrument_unreliable"]]
    AT, P, J = R["axis_table"], R["predictions"], R["join"]
    POL = [a for a in AT if a not in ("fair_month", "sign_color")]
    fig = plt.figure(figsize=(13.2, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.95, 1.5], wspace=0.42, left=0.05, right=0.985, top=0.80, bottom=0.17)
    # ---- A: spin budget
    ax = fig.add_subplot(gs[0, 0])
    keys = [("order_F3", "option order\n(4 orders)"), ("forcing", "forcing level\n(none / 3 / 5)"), ("numbering", "numbering\n(pinned vs listed)"), ("paraphrase_4", "paraphrase\n(4 stems)")]
    cols = [BLUE, ORANGE, PURPLE, GREEN]
    for i, (k, lab) in enumerate(keys):
        vals = [models[t]["spin"][k] for t in rel if models[t]["spin"].get(k) is not None]
        ax.bar(i, st.mean(vals), 0.62, color=cols[i], alpha=0.9)
        ax.scatter(i + (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.3, vals, s=12, color=INK, alpha=0.45, zorder=3, lw=0)
        ax.text(i, st.mean(vals) + 0.06, f"{st.mean(vals):.2f}", ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys], fontsize=8.2)
    ax.set_ylabel("mean range of an item's stated score\n(Likert units, −2 … +2)", fontsize=8.8, color=INK)
    ax.set_title("A  Re-ordering the options spins the answer\n    more than rewording does", fontsize=10, loc="left", fontweight="bold", color=INK)
    ax.set_ylim(0, max(2.4, ax.get_ylim()[1]))
    # ---- B: funnel
    ax = fig.add_subplot(gs[0, 1])
    counts = {k: 0 for k in ("called", "+order", "+forcing", "survives all", "refusal-gated", "not called", "saturated")}
    for t in rel:
        for a in POL:
            r = models[t]["axes"].get(a, {}).get("rung")
            if r:
                counts[r] += 1
    n_cells = sum(counts.values())
    called = counts["called"] + counts["+order"] + counts["+forcing"] + counts["survives all"]
    steps = [("model × issue cells", n_cells, GREY), ("called at the paper's\nforced format", called, "#c9d7ee"), ("direction holds across\nall orders & numberings", counts["+order"] + counts["survives all"], "#8fb3e3"),
             ("… and across forcing\n+ the “No opinion” out", counts["+forcing"] + counts["survives all"], "#5f94d6"), ("survives everything", counts["survives all"], "#1f5fb8")]
    for i, (lab, n, col) in enumerate(steps):
        ax.barh(-i, n, 0.68, color=col)
        ax.text(n + 6, -i, f"{n}" + (f"  ({100 * n / n_cells:.0f}%)" if i else ""), va="center", fontsize=9, color=INK, fontweight="bold" if i == len(steps) - 1 else "normal")
    ax.set_yticks([-i for i in range(len(steps))]); ax.set_yticklabels([s[0] for s in steps], fontsize=8.2)
    ax.set_xlim(0, n_cells * 1.38); ax.set_xticks([])
    ax.set_title("B  A paired directional readout survives\n    (difference between an issue's two statements)", fontsize=10, loc="left", fontweight="bold", color=INK)
    for s_ in ("top", "right", "bottom"):
        ax.spines[s_].set_visible(False)
    # ---- C: per-issue survival
    ax = fig.add_subplot(gs[0, 2])
    order = sorted(POL + ["fair_month", "sign_color"], key=lambda a: -(AT[a]["R3_frac"] or 0))
    ys = np.arange(len(order))
    for y, a in zip(ys, order):
        fr = AT[a]["R3_frac"] or 0
        col = GREY if a in ("fair_month", "sign_color") else CLASS_COL.get(AT[a]["class"], GREY)
        ax.barh(y, fr, 0.7, color=col)
        ax.text(fr + 0.015, y, f"{AT[a]['survives all']}/{AT[a]['n']}", va="center", fontsize=7.6, color=MUTED)
    ax.set_yticks(ys); ax.set_yticklabels([LAB.get(a, a.replace("_", " ")) for a in order], fontsize=8.2)
    ax.invert_yaxis()
    ax.axvline(2 / 3, color=MUTED, lw=0.8, ls=":"); ax.axvline(1 / 3, color=MUTED, lw=0.8, ls=":")
    ax.text(2 / 3, -0.9, "hard survivor ≥ 2/3", ha="center", fontsize=7.4, color=MUTED); ax.text(1 / 3, -0.9, "spinner < 1/3", ha="center", fontsize=7.4, color=MUTED)
    ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "", "0.5", "", "1"])
    ax.set_xlabel(f"share of {len(rel)} models whose stated direction survives everything", fontsize=8.6, color=INK)
    ax.set_title("C  Which topics survive, and survival predicts validity", fontsize=10, loc="left", fontweight="bold", color=INK)
    # inset: rho surviving vs rest
    ins = ax.inset_axes([0.60, 0.47, 0.37, 0.40])
    rs = [(j["rho_surviving"], j["rho_rest"]) for j in J.values() if j["rho_surviving"] is not None and j["rho_rest"] is not None]
    for a_, b_ in rs:
        ins.plot([0, 1], [b_, a_], "-", color=GREY, lw=0.8, alpha=0.7)
        ins.plot([0], [b_], "o", color=ORANGE, ms=3.5); ins.plot([1], [a_], "o", color=BLUE, ms=3.5)
    ins.plot([0, 1], [st.mean(b for _, b in rs), st.mean(a for a, _ in rs)], "-", color=INK, lw=2.2)
    ins.set_xticks([0, 1]); ins.set_xticklabels(["other\nissues", "surviving\nissues"], fontsize=7)
    ins.set_ylabel("ρ stated vs revealed\n(within model)", fontsize=6.8, color=INK)
    ins.tick_params(labelsize=6.5, colors=MUTED)
    ins.set_title(f"mean ρ {st.mean(b for _, b in rs):.2f} → {st.mean(a for a, _ in rs):.2f}  (p = {P['A6_permutation']['p_one_sided']:.3f})", fontsize=7, color=INK)
    for s_ in ("top", "right"):
        ins.spines[s_].set_visible(False)
    for a_ in fig.axes:
        a_.yaxis.grid(False); a_.xaxis.grid(False)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            a_.spines[s_].set_color(GRID)
        a_.tick_params(colors=MUTED)
    fig.suptitle("Re-ordering the options spins LLMs' stated opinions, but a fixed set of topics survives — and those are the ones the conjoint sees", fontsize=12, x=0.01, ha="left", fontweight="bold", color=INK)
    fig.text(0.01, 0.015, f"{len(models)} open models 1B–32B ({len(rel)} instrument-reliable), Political-Compass-style forced-choice survey (Röttger et al. 2024 format) with the option order, numbering, forcing level and an explicit opt-out varied; "
             "34 opposing-pair statements on 17 issues\nplus two valence-free null issues (grey). A: mean over models of the range of an item's score across each manipulation (dots = models). B: pre-registered ladder over model × issue cells "
             "(experiments/orderforce/PLAN.md).\nC: share of models per issue reaching the top rung; inset: Spearman correlation between the stated direction and the conjoint's revealed lean, per model, on surviving vs other issues (permutation test over survival labels).",
             fontsize=7.3, color=MUTED, va="bottom")
    out = ROOT / "experiments" / "orderforce_figure.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
