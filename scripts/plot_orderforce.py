"""Summary figure for the order/forcing experiment (PLAN.md): four equal panels, one result each.

A  Spin budget: mean range of an item's raw stated score under each manipulation, 95% CI over models (bootstrap).
B  Ladder: the model x issue cells partitioned by the highest rung reached (stacked bar, counts and shares).
C  Per-issue survival: share of reliable models whose paired direction survives every perturbation, 95% CI over models.
D  Validity: per model, Spearman rho between stated direction and revealed lean on surviving vs other issues.
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
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "orderforce.json" if (ROOT / "results" / "orderforce.json").exists() else ROOT / "experiments" / "pct" / "analysis" / "orderforce.json"
OUT = ROOT / "results" / "orderforce_figure.png" if (ROOT / "results").exists() else ROOT / "experiments" / "orderforce_figure.png"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8884", "#6250d6"
LAB = {"state_surveillance": "state surveillance", "trans_sports": "trans athletes", "housing_zoning": "housing / zoning", "agi_limitations": "AI regulation", "minimum_wage": "minimum wage",
       "police_funding": "police funding", "property_tax": "property tax", "fair_month": "fair month (null)", "sign_color": "sign colour (null)"}
NULL = ["fair_month", "sign_color"]


def boot_ci(vals, reps=4000, seed=3):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    b = np.sort([rng.choice(v, len(v)).mean() for _ in range(reps)])
    return v.mean(), b[int(0.025 * reps)], b[int(0.975 * reps) - 1]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main():
    R = json.loads(SRC.read_text())
    models = R["models"]; rel = sorted(t for t in models if not models[t]["instrument_unreliable"])
    AT, P = R["axis_table"], R["predictions"]
    POL = [a for a in AT if a not in NULL]
    a6 = P["A6_permutation"]
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.6), facecolor=SURF)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.915, bottom=0.115, wspace=0.40, hspace=0.45)
    # ---- A spin budget
    ax = axs[0][0]
    keys = [("order_F3", "option order\n(4 orders)", BLUE), ("forcing", "forcing level\n(none / 3 / 5)", ORANGE), ("numbering", "numbering\n(pinned vs listed)", PURPLE), ("paraphrase_4", "paraphrase\n(4 stems)", GREEN)]
    for i, (k, lab, col) in enumerate(keys):
        vals = [models[t]["spin"][k] for t in rel if models[t]["spin"].get(k) is not None]
        m, lo, hi = boot_ci(vals)
        ax.scatter([i + x for x in np.linspace(-0.18, 0.18, len(vals))], sorted(vals), s=14, color=GREY, alpha=0.6, lw=0, zorder=2)
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=8, capsize=5, lw=1.6, mec=SURF, mew=1, zorder=4)
        ax.text(i + 0.26, m, f"{m:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k[1] for k in keys], fontsize=8.5)
    ax.set_xlim(-0.6, len(keys) - 0.3); ax.set_ylim(0, None)
    ax.set_ylabel("mean range of an item's stated score\n(Likert units, −2 … +2)", fontsize=9, color=INK)
    ax.set_title(f"A   Spin of the raw stated score under each manipulation\n      mean ± 95% CI over {len(rel)} models; dots = models", fontsize=9.5, loc="left", fontweight="bold", color=INK)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    # ---- B ladder partition
    ax = axs[0][1]
    counts = {k: 0 for k in ("not called", "refusal-gated", "called", "+order", "+forcing", "survives all", "saturated")}
    for t in rel:
        for a in POL:
            r = models[t]["axes"].get(a, {}).get("rung")
            if r:
                counts[r] += 1
    n = sum(counts.values())
    parts = [("no called direction at the paper's format", counts["not called"] + counts["refusal-gated"] + counts["saturated"], "#d9d7d2"),
             ("called, lost under both kinds of change", counts["called"], "#c9d7ee"), ("holds across forcing + opt-out only", counts["+forcing"], "#8fb3e3"),
             ("holds across orders + numberings only", counts["+order"], "#5f94d6"), ("holds across every perturbation", counts["survives all"], "#1f5fb8")]
    for i, (lab, c, col) in enumerate(parts):
        ax.barh(i, c, 0.62, color=col)
        ax.text(c + 3, i, f"{c}  ({100 * c / n:.0f}%)", va="center", fontsize=8.8, color=INK)
    ax.set_yticks(range(len(parts))); ax.set_yticklabels([p[0] for p in parts], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, n * 1.15); ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel(f"model × issue cells (n = {n}: {len(rel)} models × 17 issues)", fontsize=9, color=INK)
    ax.set_title("B   Highest rung reached by each model × issue cell\n      (paired direction = an issue's agree − disagree difference)", fontsize=9.5, loc="left", fontweight="bold", color=INK)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    # ---- C per-issue survival with CI over models
    ax = axs[1][0]
    order = sorted(POL, key=lambda a: -(AT[a]["R3_frac"] or 0)) + NULL
    for y, a in enumerate(order):
        surv = [1.0 if models[t]["axes"].get(a, {}).get("rung") == "survives all" else 0.0 for t in rel if a in models[t]["axes"]]
        m, lo, hi = boot_ci(surv)
        col = GREY if a in NULL else GREEN if AT[a]["class"] == "hard survivor" else BLUE if AT[a]["class"] == "mixed" else ORANGE
        ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=6, capsize=3, lw=1.2, mec=SURF, mew=0.7, zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB.get(a, a.replace("_", " ")) for a in order], fontsize=8.5)
    ax.invert_yaxis()
    for x, lab in ((1 / 3, "1/3"), (2 / 3, "2/3")):
        ax.axvline(x, color=GRID, lw=1, ls="--", zorder=0)
    ax.set_xlim(-0.03, 1.08); ax.set_xticks([0, 1 / 3, 2 / 3, 1]); ax.set_xticklabels(["0", "1/3", "2/3", "1"])
    ax.set_xlabel(f"share of {len(rel)} models whose paired direction survives every perturbation (± 95% CI)", fontsize=9, color=INK)
    ax.set_title("C   Survival by issue\n      green ≥ 2/3 of models, blue 1/3–2/3, orange < 1/3, grey = null controls", fontsize=9.5, loc="left", fontweight="bold", color=INK)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    # ---- D validity scatter
    ax = axs[1][1]
    J = R["join"]
    pts = [(j["rho_rest"], j["rho_surviving"], t) for t, j in J.items() if t in rel and j["rho_surviving"] is not None and j["rho_rest"] is not None and j["n_surviving"] >= 3 and j["n"] - j["n_surviving"] >= 3]
    ax.plot([-1, 1], [-1, 1], color=GRID, lw=1, zorder=0)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=BLUE, ec=SURF, lw=0.8, zorder=3)
    ax.scatter([a6["mean_rho_rest"]], [a6["mean_rho_surviving"]], s=120, marker="D", color=ORANGE, ec=SURF, lw=1, zorder=4, label=f"mean ({a6['mean_rho_rest']:.2f}, {a6['mean_rho_surviving']:.2f})")
    lo_, hi_ = min(min(p[0] for p in pts), min(p[1] for p in pts)) - 0.1, 1.02
    ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_); ax.set_aspect("equal")
    ax.set_xlabel("Spearman ρ on the other issues", fontsize=9, color=INK); ax.set_ylabel("Spearman ρ on the surviving issues", fontsize=9, color=INK)
    ax.set_title(f"D   Agreement with the revealed lean, by survival\n      one point per model (n = {len(pts)})", fontsize=9.5, loc="left", fontweight="bold", color=INK)
    ax.text(0.03, 0.97, f"above the diagonal: {sum(p[1] > p[0] for p in pts)} of {len(pts)}\nmean gain {a6['mean_gain_rho']:+.2f}, permutation p = {a6['p_one_sided']:.3f}", transform=ax.transAxes, fontsize=8.3, color=INK, va="top")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.xaxis.grid(True, color=GRID); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for row in axs:
        for a_ in row:
            style(a_)
    fig.suptitle("Which stated preferences survive re-ordering the options and forcing the answer?", fontsize=12.5, x=0.115, y=0.975, ha="left", fontweight="bold", color=INK)
    fig.text(0.115, 0.012, f"{len(models)} open models, 1B–32B ({len(rel)} instrument-reliable). Political-Compass-style forced-choice survey in the Röttger et al. (2024) format, with the option order\n"
             "(4 orders), the numbering (list-following vs label-pinned), the forcing level (none / label-only / 3 / 5) and an explicit “No opinion” option varied; 34 opposing-pair\n"
             "statements on 17 issues plus two valence-free null issues; readout = teacher-forced label probabilities. A paired direction is the agree−disagree difference between\n"
             "an issue's two statements. Rungs (B, C) and the revealed lean (D, from an independent candidate-choice conjoint) are defined in PLAN.md; intervals are bootstraps over models.",
             fontsize=7.4, color=MUTED, va="bottom")
    fig.savefig(OUT, dpi=170, facecolor=SURF)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
