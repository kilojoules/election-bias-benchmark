"""Summary figure for the order/forcing experiment (PLAN.md): a survival ledger.

Band A: how far each manipulation moves an item's raw stated score (per-model circles, mean diamonds).
Main panel B: per issue, the share of reliable models whose paired direction is called (grey track), lost to a
perturbation (orange) and survives everything (dot, coloured by class); null-control issues as open circles.
Inset C: within-model Spearman rho between the stated direction and the revealed lean, surviving vs other issues.
Reads results/orderforce.json (standalone) or experiments/pct/analysis/orderforce.json (parent repo).
Usage: python3 scripts/plot_orderforce.py
"""
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "orderforce.json" if (ROOT / "results" / "orderforce.json").exists() else ROOT / "experiments" / "pct" / "analysis" / "orderforce.json"
OUT = ROOT / "results" / "orderforce_figure.png" if (ROOT / "results").exists() else ROOT / "experiments" / "orderforce_figure.png"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8884"
CLASS_COL = {"hard survivor": GREEN, "mixed": BLUE, "spinner": GREY}
LAB = {"state_surveillance": "state surveillance", "trans_sports": "trans athletes", "housing_zoning": "housing / zoning", "agi_limitations": "AI regulation", "minimum_wage": "minimum wage",
       "police_funding": "police funding", "property_tax": "property tax", "fair_month": "county fair month *", "sign_color": "road-sign colour *"}


def main():
    R = json.loads(SRC.read_text())
    models = R["models"]; rel = sorted(t for t in models if not models[t]["instrument_unreliable"])
    AT, P, J = R["axis_table"], R["predictions"], R["join"]
    NULL = ["fair_month", "sign_color"]; POL = [a for a in AT if a not in NULL]
    counts = {k: 0 for k in ("called", "+order", "+forcing", "survives all", "refusal-gated", "not called", "saturated")}
    for t in rel:
        for a in POL:
            r = models[t]["axes"].get(a, {}).get("rung")
            if r:
                counts[r] += 1
    n_cells = sum(counts.values()); called = counts["called"] + counts["+order"] + counts["+forcing"] + counts["survives all"]
    n_order, n_forc, n_all = counts["+order"] + counts["survives all"], counts["+forcing"] + counts["survives all"], counts["survives all"]
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(11.0, 7.6), facecolor=SURF)
    # ---------------- band A
    axA = fig.add_axes([0.30, 0.805, 0.56, 0.105])
    rows = [("order_F3", "option order", BLUE), ("forcing", "forcing level", ORANGE), ("numbering", "numbering", GREY), ("paraphrase_4", "rewording", GREEN)]
    jit = np.linspace(-0.10, 0.10, len(rel))
    means = {}
    for i, (k, lab, col) in enumerate(rows):
        y = len(rows) - 1 - i
        vals = [models[t]["spin"].get(k) for t in rel]
        vals = [v for v in vals if v is not None]
        axA.scatter(vals, [y + j for j in jit[:len(vals)]], s=16, facecolors="none", edgecolors=GREY, lw=0.8, alpha=0.75, zorder=2)
        m = st.mean(vals); means[k] = m
        axA.scatter([m], [y], marker="D", s=42, color=col, zorder=4, ec=SURF, lw=0.6)
        axA.text(m + 0.05, y, f"{m:.2f}" + ("   %.0f× the rewording effect" % (m / means.get("paraphrase_4", m)) if k == "order_F3" and "paraphrase_4" in means else ""), va="center", fontsize=8.5, color=INK, fontweight="bold" if k == "order_F3" else "normal")
    # order row is drawn first; recompute its multiplier label after the rewording mean is known
    axA.texts[0].set_text(f"{means['order_F3']:.2f}   {means['order_F3'] / means['paraphrase_4']:.0f}× the rewording effect")
    axA.axvline(means["paraphrase_4"], color=MUTED, lw=0.8, ls="--", alpha=0.45, zorder=1)
    axA.text(means["paraphrase_4"] + 0.02, -0.5, "rewording mean", fontsize=7.0, color=MUTED, ha="left", va="center")
    axA.set_yticks(range(len(rows))); axA.set_yticklabels([r[1] for r in reversed(rows)], fontsize=8.5, color=MUTED)
    axA.set_xlim(0, max(2.0, max(models[t]["spin"]["order_F3"] for t in rel) + 0.15)); axA.set_ylim(-0.8, len(rows) - 0.3)
    axA.set_xticks([0, 1, 2]); axA.tick_params(colors=MUTED, labelsize=8, length=0)
    axA.xaxis.grid(True, color=GRID, lw=0.8); axA.set_axisbelow(True)
    axA.set_xlabel(f"mean range of an item's stated score (Likert −2…+2), {len(rel)} reliable models", fontsize=8, color=MUTED, labelpad=2)
    axA.set_title("What the raw stated score does when the paper's fixed choices vary   (circles = models, diamonds = means)", fontsize=8.8, loc="left", fontweight="bold", color=INK, pad=5)
    # ---------------- main panel B
    axB = fig.add_axes([0.30, 0.175, 0.56, 0.50])
    order = sorted(POL, key=lambda a: -(AT[a]["R3_frac"] or 0)) + NULL
    for y, a in enumerate(reversed(order)):
        r3, cf = AT[a]["R3_frac"] or 0.0, AT[a]["called_frac"] or 0.0
        axB.plot([0, cf], [y, y], color=GRID, lw=2.2, solid_capstyle="butt", zorder=1)
        if cf > r3:
            axB.plot([r3, cf], [y, y], color=ORANGE, lw=2.2, solid_capstyle="butt", zorder=2)
        if a in NULL:
            axB.scatter([r3], [y], s=30, facecolors=SURF, edgecolors=GREY, lw=1.1, zorder=4)
        else:
            axB.scatter([r3], [y], s=30, color=CLASS_COL.get(AT[a]["class"], GREY), ec=SURF, lw=0.6, zorder=4)
        if r3 >= 0.999 or a in NULL:
            axB.text(max(cf, r3) + 0.025, y, f"{AT[a]['survives all']}/{AT[a]['n']}", va="center", fontsize=7.4, color=INK, fontweight="bold" if r3 >= 0.999 else "normal")
    axB.set_yticks(range(len(order))); axB.set_yticklabels([LAB.get(a, a.replace("_", " ")) for a in reversed(order)], fontsize=8.3)
    for lab, a in zip(axB.get_yticklabels(), reversed(order)):
        if a in NULL:
            lab.set_color(GREY); lab.set_style("italic")
    for x in (1 / 3, 2 / 3):
        axB.axvline(x, color=GRID, lw=1, ls="--", zorder=0)
        axB.text(x, -1.05, "1/3" if x < 0.5 else "2/3", fontsize=7, color=MUTED, ha="center", va="top")
    axB.set_xlim(0, 1.14); axB.set_ylim(-1.3, len(order) - 0.4)
    axB.set_xticks([0, 0.25, 0.5, 0.75, 1]); axB.tick_params(colors=MUTED, labelsize=8, length=0)
    axB.set_xlabel(f"share of the {len(rel)} reliable models whose paired direction survives every perturbation", fontsize=8.5, color=INK)
    axB.text(0.22, 0.5, f"← null controls: {R['null_called']} of {2 * len(rel)} cells called", fontsize=7, color=MUTED, ha="left", va="center")
    axB.text(0.0, -0.115, f"{n_cells} model × issue cells → {called} with a called direction → {n_order} stable across all 4 orders + both numberings, {n_forc} across forcing + the opt-out\n"
             f"(parallel criteria: {n_order - n_all} order-only, {n_forc - n_all} forcing-only) → {n_all} survive everything ({100 * n_all / n_cells:.0f}%)",
             transform=axB.transAxes, fontsize=7.4, color=MUTED, va="top")
    # ---------------- inset C
    axC = fig.add_axes([0.655, 0.205, 0.20, 0.10])
    a6 = P["A6_permutation"]
    ms, mr = a6["mean_rho_surviving"], a6["mean_rho_rest"]
    axC.set_facecolor(SURF)
    axC.plot([mr, ms], [1, 0], color=GREY, lw=1.2, zorder=1)
    axC.scatter([ms], [0], s=30, color=GREEN, zorder=3, ec=SURF, lw=0.5); axC.scatter([mr], [1], s=30, color=GREY, zorder=3, ec=SURF, lw=0.5)
    axC.text(ms + 0.03, 0, f"ρ = {ms:.2f}", va="center", fontsize=8, color=GREEN, fontweight="bold"); axC.text(mr + 0.03, 1, f"ρ = {mr:.2f}", va="center", fontsize=8, color=INK)
    axC.set_yticks([0, 1]); axC.set_yticklabels(["surviving issues", "other issues"], fontsize=7.5, color=MUTED)
    axC.set_xlim(0, 1.05); axC.set_ylim(-0.7, 1.7); axC.set_xticks([0, 0.5, 1]); axC.tick_params(colors=MUTED, labelsize=7, length=0)
    axC.set_title(f"Survival predicts validity\nstated vs revealed lean, {a6['models']} models, p = {a6['p_one_sided']:.3f}", fontsize=7.8, fontweight="bold", color=INK, loc="left", pad=3)
    for a_ in (axA, axB, axC):
        for s_ in ("top", "right", "left"):
            a_.spines[s_].set_visible(False)
        a_.spines["bottom"].set_color(GRID)
    # ---------------- legend, title, caption
    handles = [Line2D([], [], marker="o", color=GREEN, lw=0, ms=6, label="survives all perturbations (≥ 2/3 of models)"), Line2D([], [], marker="o", color=BLUE, lw=0, ms=6, label="mixed (1/3–2/3)"),
               Line2D([], [], marker="o", color=GREY, lw=0, ms=6, label="spins (< 1/3)"), Line2D([], [], color=ORANGE, lw=2.2, label="called, lost to a perturbation"),
               Line2D([], [], color=GRID, lw=2.2, label="share with a called direction"), Line2D([], [], marker="o", mfc=SURF, mec=GREY, color=SURF, lw=0, ms=6, label="null-control issue *")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.295, 0.735), ncol=3, frameon=False, fontsize=7.4, handletextpad=0.5, columnspacing=1.4)
    fig.text(0.02, 0.965, f"Re-ordering the options spins stated answers {means['order_F3'] / means['paraphrase_4']:.0f}× more than rewording; paired directions survive, topic by topic", fontsize=12, fontweight="bold", color=INK, va="center")
    fig.text(0.02, 0.075, f"{len(models)} open LLMs ({len(rel)} instrument-reliable) answered Political-Compass-style opposing-pair items on 17 issues plus 62 PCT propositions in the Röttger et al. forced-choice format while we varied what that format holds\n"
             "fixed: option order (4), numbering (list-following vs label-pinned), forcing level (none / label-only / 3 / 5) and an explicit “No opinion” option; readout = teacher-forced label probabilities. Band: mean within-item\n"
             "range of the raw stated score; circles = reliable models, diamonds = means. Main: the paired directional readout (an issue's two opposing statements) must keep its sign under every perturbation; the two\n"
             "stability criteria are parallel, not nested; * = invented null-control issues. Inset: within-model Spearman ρ between the stated direction and the independently measured revealed lean. Pre-registered: PLAN.md.",
             fontsize=7.2, color=MUTED, va="top")
    fig.savefig(OUT, dpi=190, facecolor=SURF)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
