"""Figures for the expanded sweep, from experiments/sweep/summary.json (scripts/sweep_analyze.py).

sweep_heatmap.png      models x flipped issues, focal = property tax (the continuity cell), signed win-rate
sweep_heatmap_all.png  the same for all three focal issues side by side
sweep_scale.png        called leans vs parameter count, families with >= 3 sizes connected
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
ISSUES = json.loads((ROOT / "data" / "issues.json").read_text())
ROSTER = json.loads((ROOT / "scripts" / "sweep_models.json").read_text())
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
# diverging: blue (prefers the other level) <- neutral gray -> red (prefers the report level); equal steps per arm
CMAP = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#2a78d6", "#6da7ec", "#b7d3f6", "#f0efec", "#f3b9b8", "#ec8382", "#e34948", "#b12e2e"])
VLIM = 0.3
LOLIM = 2.0   # nats, for the log-odds heatmap
FAMILY_SLOTS = {"Gemma 3": "#2a78d6", "Gemma 4": "#eb6834", "Qwen3.5": "#1baf7a", "Qwen3": "#eda100", "Llama 3": "#e87ba4"}


def model_order(res):
    fam_order = []
    for m in ROSTER["models"]:
        if m.get("family") not in fam_order:
            fam_order.append(m.get("family"))
    tags = [t for t in res]
    def key(t):
        r = res[t]
        return (fam_order.index(r["family"]) if r["family"] in fam_order else 99, r["params_b"] or 0)
    return sorted(tags, key=key)


def label(t, r):
    p = r["params_b"]
    return f"{t}" if not p else f"{t}"


def heat_panel(ax, res, tags, focal, cols, title, show_ylabels=True, readout="rate"):
    vlim = VLIM if readout == "rate" else LOLIM
    norm = Normalize(-vlim, vlim)
    for i, t in enumerate(tags):
        cells = res[t]["cells"]
        for j, a in enumerate(cols):
            c = cells.get(f"{focal}|{a}")
            if not c:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor="white", edgecolor=GRID, lw=0.5))
                continue
            v = (c["rate"] - 0.5) if readout == "rate" else c["lo_mean"]
            lean = c["lean"] if readout == "rate" else c["lo_lean"]
            cell_label = f"{c['rate']:.2f}" if readout == "rate" else f"{c['lo_mean']:+.1f}"
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=CMAP(norm(v)), edgecolor=SURF, lw=1.5))
            if lean:
                ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, fill=False, edgecolor=INK, lw=1.6))
            txt = INK if abs(v) < 0.65 * vlim else "#fcfcfb"
            ax.text(j + 0.5, i + 0.5, cell_label, ha="center", va="center", fontsize=6.2, color=txt,
                    fontweight="bold" if lean else "normal")
    n_pol = sum(1 for a in cols if not ISSUES["axes"][a].get("null"))
    ax.axvline(n_pol, color=INK, lw=1.0)
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(tags), 0)
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels([f"{ISSUES['axes'][a]['topic']}\n→ {ISSUES['axes'][a]['poles'][ISSUES['axes'][a]['report']]}" for a in cols],
                       rotation=90, fontsize=6.4, color=INK)
    ax.set_yticks([i + 0.5 for i in range(len(tags))])
    if show_ylabels:
        flag = (lambda r: r["null_axis_leans"]) if readout == "rate" else (lambda r: r.get("null_axis_lo_leans"))
        ax.set_yticklabels([label(t, res[t]) + (" †" if res[t]["low_mass"] else "") + (" ‡" if flag(res[t]) else "") for t in tags], fontsize=6.6, color=INK)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def heatmap(res, tags, focals, path, suptitle, readout="rate"):
    cols = [a for a in ISSUES["axes"] if not ISSUES["axes"][a].get("null")] + [a for a in ISSUES["axes"] if ISSUES["axes"][a].get("null")]
    cols = [a for a in cols if a not in focals] if len(focals) == 1 else cols
    w = 0.36 * len(cols) * len(focals) + 3.2
    h = 0.22 * len(tags) + 3.6
    fig, axes = plt.subplots(1, len(focals), figsize=(w, h), sharey=False, layout="constrained")
    axes = [axes] if len(focals) == 1 else list(axes)
    for k, (ax, focal) in enumerate(zip(axes, focals)):
        cols_f = [a for a in cols if a != focal]
        heat_panel(ax, res, tags, focal, cols_f, f"voter's only concern: {ISSUES['axes'][focal]['topic']}", show_ylabels=(k == 0), readout=readout)
    vlim = VLIM if readout == "rate" else LOLIM
    sm = plt.cm.ScalarMappable(norm=Normalize(-vlim, vlim), cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, aspect=40)
    cb.set_label("win-rate − 0.5" if readout == "rate" else "mean log-odds (nats)", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK)
    cb.outline.set_visible(False)
    fig.suptitle(suptitle, fontsize=10.5, color=INK)
    if readout == "rate":
        foot = ("Cell = P(model names the candidate holding the column's → level) when both candidates share the focal issue and two filler issues and differ only on that issue; red = names it more often, blue = less.\n"
                "Outlined bold cells: called leans (bootstrap CI excludes 0.5, |effect| ≥ 0.05, BH q ≤ 0.01 within model). Right of the black line: valence-free null axes.\n"
                "† mean naming mass < 0.2 (not a forced choice for this model)   ‡ lean on a null axis (instrument-unreliable)")
    else:
        foot = ("Cell = mean of log p(→ candidate's name) − log p(other name) over the same trials, clipped to ±12 nats; positive = favours the → candidate. Unlike the win-rate this does not\n"
                "saturate when a model names the first-listed candidate with P ≈ 1. Outlined bold cells: CI excludes 0, |mean| ≥ 0.3 nats, BH q ≤ 0.01 within model. Not preregistered; see PLAN.md.\n"
                "Right of the black line: valence-free null axes.   † mean naming mass < 0.2   ‡ log-odds lean on a null axis (readout unreliable for that model)")
    fig.supxlabel(foot, fontsize=6.6, color=MUTED)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("wrote", path)


def scale_plot(res, tags, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    fams = {}
    for t in tags:
        r = res[t]
        if r["params_b"]:
            fams.setdefault(r["family"], []).append((r["params_b"], r["n_leans"], t, r["low_mass"] or bool(r["null_axis_leans"])))
    for fam, pts in fams.items():
        pts.sort()
        col = FAMILY_SLOTS.get(fam, "#c3c2b7")
        if fam in FAMILY_SLOTS and len(pts) >= 3:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=1.6, zorder=2, label=fam)
        for x, y, t, bad in pts:
            ax.plot(x, y, "o", color=col, ms=7 if fam in FAMILY_SLOTS else 5, mec=SURF, mew=1, zorder=3, alpha=0.5 if bad else 1)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (B, total)", fontsize=9.5, color=INK)
    ax.set_ylabel("called leans (of 51 political cells)", fontsize=9.5, color=INK)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, title="families with ≥ 3 sizes", title_fontsize=8)
    ax.set_title("Issue-contingent candidate preference does not vanish with scale", fontsize=10.5, color=INK, loc="left")
    fig.text(0.99, 0.01, "faded: low naming mass or a lean on a null axis", fontsize=7, color=MUTED, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", path)


def position_plot(res, tags, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for t in tags:
        r = res[t]
        pe = r.get("pos_effect_abs", r.get("pos_effect"))
        if not r["params_b"] or pe is None:
            continue
        col = FAMILY_SLOTS.get(r["family"], "#c3c2b7")
        ax.plot(r["params_b"], pe, "o", color=col, ms=7 if r["family"] in FAMILY_SLOTS else 5, mec=SURF, mew=1,
                alpha=0.45 if r["low_mass"] else 1, zorder=3)
        if pe >= 0.9 or pe <= 0.15:
            ax.annotate(t, (r["params_b"], pe), fontsize=6, color=MUTED, xytext=(4, 3), textcoords="offset points")
    for fam, col in FAMILY_SLOTS.items():
        if any(res[t]["family"] == fam for t in tags):
            ax.plot([], [], "o", color=col, label=fam)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("parameters (B, total)", fontsize=9.5, color=INK)
    ax.set_ylabel("listing-position effect, mean over the three focal framings of\n|P(names X | X listed first) − P(names X | X listed second)|", fontsize=9, color=INK)
    ax.set_title("Most models above ~10B answer by listing order; the win-rate readout saturates there", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.99, 0.01, "faded: naming mass < 0.2", fontsize=7, color=MUTED, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", path)


def main():
    d = ROOT / "experiments" / "sweep"
    res = json.loads((d / "summary.json").read_text())
    tags = model_order(res)
    heatmap(res, tags, ["property_tax"], d / "sweep_heatmap.png",
            "Which issues sway which models? Forced choice between two candidates with identical property-tax positions")
    heatmap(res, tags, ISSUES["focal_axes"], d / "sweep_heatmap_all.png",
            "The same readout for three different focal issues (what the voter says they care about)")
    heatmap(res, tags, ["property_tax"], d / "sweep_heatmap_logodds.png",
            "Exploratory log-odds readout (does not saturate when a model picks by listing position), focal = property tax", readout="logodds")
    scale_plot(res, tags, d / "sweep_scale.png")
    position_plot(res, tags, d / "sweep_position.png")


if __name__ == "__main__":
    main()
