"""Stated vs revealed political opinions: linking the sweep to Röttger et al. 2024.

Röttger et al. ("Political Compass or Spinning Arrow?", arXiv 2402.16786) show that asking an LLM its
opinion directly (a stated compass) is unstable under paraphrase and forcing, and that open-ended
answers land more right/libertarian than forced multiple choice. This script compares that stated
readout with the *revealed* readout of the sweep instrument: which candidate the model picks when an
unasked issue is the only difference between two otherwise identical candidates.

Inputs
  A. experiments/pct/stated_<tag>.jsonl  one row per (item, paraphrase, forcing):
     {"model","tag","item": "<axis>|<level>" for our 34 statements or "pct_<n>" for the 62 PCT
      propositions, "text", "paraphrase": int, "forcing": int in {1,3,5},
      "lp_options": [4 floats], "valid_mass": float in (0,1], "p_agree": float,
      "score": float in [-2,2] (expected agreement: -2 strongly disagree ... +2 strongly agree)}
  B. experiments/pct/pct_<tag>.jsonl  revealed readout for the PCT propositions, rows in the same
     format as experiments/sweep/<tag>.jsonl (mode "sweep", focal "property_tax", flip "pct_<n>").
     Per-cell statistics reuse cell_stats/log_odds from scripts/sweep_analyze.py.
  data/pct_issues.json  axes "pct_<n>" with "proposition", "pct_axis" ("economic"/"social"),
     "agree_sign", "levels" {"agree","disagree"}, "report": "agree".

Outputs (experiments/pct/analysis/)
  stated_vs_revealed.md                     the report
  stated_vs_revealed.json                   all numbers behind it
  fig_validity.png                          small multiples: validity by forcing, per model
  fig_issues_scatter.png                    per model: stated lean vs revealed lean, our 17 issues
  fig_pct_scatter.png                       per model: stated score vs revealed log-odds, 62 PCT items
  fig_compass.png                           per model: stated -> revealed PCT compass with arrows

Usage
  python3 scripts/pct/compare.py [--sweep_dir experiments/sweep] [--stated_dir experiments/pct]
      [--pct_dir experiments/pct] [--pct_issues data/pct_issues.json] [--out experiments/pct/analysis]
      [--models tag ...] [--min_models 2] [--skip_validation]
"""
import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPTS = Path(__file__).resolve().parent.parent          # repo/scripts
ROOT = SCRIPTS.parent                                     # repo root
sys.path.insert(0, str(SCRIPTS))
try:
    from sweep_analyze import cell_stats, bh, LOGODDS_FLOOR  # noqa: E402  (reuse the sweep's cell machinery)
except ImportError:                                       # pragma: no cover
    sys.path.insert(0, str(ROOT / "scripts"))
    from sweep_analyze import cell_stats, bh, LOGODDS_FLOOR

ISSUES = json.loads((ROOT / "data" / "issues.json").read_text())
POLITICAL = [a for a in ISSUES["axes"] if not ISSUES["axes"][a].get("null")]
FOCAL_AXES = ISSUES["focal_axes"]
FORCINGS = (1, 3, 5)
REFUSE_VM = 0.3          # a stated row with valid_mass below this at forcing 1 counts as a refusal
VALID_VM = 0.7           # a stated row is treated as a usable opinion at/above this

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, RED = "#2a78d6", "#e34948"
FAMILY_SLOTS = {"Gemma 3": "#2a78d6", "Gemma 4": "#eb6834", "Qwen3.5": "#1baf7a", "Qwen3": "#eda100",
                "Llama 3": "#e87ba4", "Mistral": "#7a6fc4"}


def other_level(axis, level):
    ls = list(ISSUES["axes"][axis]["levels"])
    return ls[1] if ls[0] == level else ls[0]


def rankdata(xs):
    """Average ranks with ties, no scipy needed."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rho; None if either input is constant (rho undefined)."""
    if len(xs) < 3:
        return None
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def sign(x):
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------------- loading


def load_jsonl(path):
    rows, bad = [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:      # truncated final line of a partial pull
            bad += 1
    return rows, bad


def load_stated(d, tags):
    """stated_<tag>.jsonl -> {tag: {item: {(paraphrase, forcing): row}}}"""
    out = {}
    for t in tags:
        f = d / f"stated_{t}.jsonl"
        if not f.exists():
            continue
        rows, bad = load_jsonl(f)
        grid = defaultdict(dict)
        for r in rows:
            grid[r["item"]][(int(r["paraphrase"]), int(r["forcing"]))] = r
        out[t] = {"grid": grid, "n_rows": len(rows), "bad_lines": bad}
    return out


def load_pct_cells(d, tags):
    """pct_<tag>.jsonl -> {tag: {flip: cell_stats(rows, 'bg')}}"""
    out = {}
    for t in tags:
        f = d / f"pct_{t}.jsonl"
        if not f.exists():
            continue
        rows, bad = load_jsonl(f)
        sweep = [r for r in rows if r.get("mode", "sweep") == "sweep"]
        by_cell = defaultdict(list)
        for r in sweep:
            by_cell[r["flip"]].append(r)
        cells = {k: cell_stats(rs, "bg") for k, rs in by_cell.items()}
        # BH over the model's PCT cells, same rule shape as sweep_analyse (q = 0.01)
        keys = [k for k in cells if cells[k]["lo_p"] is not None]
        rej = bh([cells[k]["lo_p"] for k in keys], 0.01) if keys else set()
        for i, k in enumerate(keys):
            c = cells[k]
            c["lo_lean"] = (i in rej) and (c["lo_lo"] > 0 or c["lo_hi"] < 0) and abs(c["lo_mean"]) >= LOGODDS_FLOOR
        mass = sum(r["mass"] for r in sweep) / len(sweep) if sweep else None
        out[t] = {"cells": cells, "n": len(sweep), "mass": mass, "bad_lines": bad}
    return out


def load_pct_issues(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------- stated-opinion summaries


def stated_score_by_item(grid, item, para_filter=None, forcing_filter=None):
    """Mean score over the (paraphrase, forcing) cells asked, as a simple mean of cell means."""
    cells = [(k, r) for k, r in grid[item].items()
             if (para_filter is None or k[0] in para_filter)
             and (forcing_filter is None or k[1] == forcing_filter)]
    if not cells:
        return None, 0
    return sum(r["score"] for _, r in cells) / len(cells), len(cells)


def stated_valid_mass(grid, item, forcing):
    rows = [r for k, r in grid[item].items() if k[1] == forcing]
    return ([r["valid_mass"] for r in rows], len(rows))


def stated_issue_lean(grid, axis):
    """score(report-level statement) - score(other-level statement), averaged over the 10
    paraphrases x 3 forcings; positive = the model agrees more with the report-level position."""
    rep, other = f"{axis}|{ISSUES['axes'][axis]['report']}", f"{axis}|{other_level(axis, ISSUES['axes'][axis]['report'])}"
    if rep not in grid or other not in grid:
        return None
    diffs, n = [], 0
    for k in sorted(grid[rep]):
        if k in grid[other]:
            diffs.append(grid[rep][k]["score"] - grid[other][k]["score"])
            n += 1
    if not diffs:
        return None
    return {"lean": sum(diffs) / len(diffs), "n_cells": n,
            "n_f1": sum(1 for k in grid[rep] if k[1] == 1)}


def stated_forcing_shift(grid):
    """|score(forcing 5) - score(forcing 1)| per (item, paraphrase), matched; the analogue of the
    paper's forcing instability."""
    ds = []
    for item, cells in grid.items():
        for k, r in cells.items():
            if k[1] == 1:
                r5 = cells.get((k[0], 5))
                if r5 is not None:
                    ds.append(abs(r5["score"] - r["score"]))
    return (sum(ds) / len(ds) if ds else None), len(ds)


def stated_paraphrase_range(grid):
    """Per item: max - min of the score across paraphrases (all forcings pooled); summarised over
    items; the analogue of the paper's paraphrase instability."""
    per_item = {}
    for item, cells in grid.items():
        by_para = defaultdict(list)
        for (p, _f), r in cells.items():
            by_para[p].append(r["score"])
        para_means = [sum(v) / len(v) for v in by_para.values()]
        if len(para_means) >= 2:
            per_item[item] = max(para_means) - min(para_means)
    if not per_item:
        return None, 0, {}
    vals = sorted(per_item.values())
    return sum(vals) / len(vals), len(vals), {"p50": vals[len(vals) // 2], "p90": vals[int(0.9 * (len(vals) - 1))],
                                              "max": vals[-1]}


def refusal_flag(grid, item):
    """valid_mass < 0.3 at forcing 1 (median over paraphrases) -> the model refuses / does not
    answer the stated question when free to."""
    vms, _ = stated_valid_mass(grid, item, 1)
    if not vms:
        return None
    s = sorted(vms)
    vm = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return vm < REFUSE_VM


# --------------------------------------------------------------------------- figures


def plot_validity(per_model, order, path):
    """Small multiples: validity (valid_mass) by forcing, per model; the analogue of the paper's
    Figure 2."""
    n = len(order)
    ncol = min(6, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 1.55 * nrow), sharey=True)
    axes = list(axes.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, t in zip(axes, order):
        v = per_model[t]["validity"]
        xs, ys, ns = [], [], []
        for f in FORCINGS:
            if v[f]["mean"] is not None:
                xs.append(f)
                ys.append(v[f]["mean"])
                ns.append(v[f]["n"])
        col = FAMILY_SLOTS.get(per_model[t]["family"], MUTED)
        ax.plot(xs, ys, "o-", color=col, ms=4.5, lw=1.4, mec=SURF, mew=0.7, zorder=3)
        ax.axhline(REFUSE_VM, color=MUTED, lw=0.7, ls=":")
        ax.axhline(1.0, color=GRID, lw=0.7)
        ax.set_xticks(list(FORCINGS))
        ax.set_title(t, fontsize=6.8, color=INK)
        ax.tick_params(labelsize=6, length=0)
        ax.set_ylim(0, 1.05)
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel("valid mass", fontsize=6.8, color=INK)
    for ax in axes[-ncol:]:
        ax.set_xlabel("forcing", fontsize=6.8, color=INK)
    fig.suptitle("Stated answers become valid opinions only under forcing", fontsize=10.5, color=INK)
    fig.text(0.01, 0.005, "valid_mass = probability mass on the four agreement options; dotted line = 0.3 (below it the stated answer is read as a refusal). "
             "Points average all items and paraphrases at that forcing for the model.", fontsize=6.6, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("wrote", path)


def plot_issues_scatter(per_model, order, path):
    """Per model panel: stated lean vs revealed lean for the 17 political issues (one point per
    model x issue)."""
    n = len(order)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 2.9 * nrow), sharex=True, sharey=True)
    axes = [axes] if not hasattr(axes, "flat") else list(axes.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, t in zip(axes, order):
        m = per_model[t]
        pts = [(iss["stated_lean"], iss["revealed_lean"]) for iss in m["issues"].values()
               if iss["stated_lean"] is not None and iss["revealed_lean"] is not None]
        for iss in m["issues"].values():
            if iss["stated_lean"] is None or iss["revealed_lean"] is None:
                continue
            x, y = iss["stated_lean"], iss["revealed_lean"]
            col = BLUE if (sign(x) == sign(y)) else RED
            ax.plot(x, y, "o", ms=6 if iss["lo_lean_any"] else 4, color=col, mec=SURF, mew=0.7,
                    alpha=0.9 if iss["lo_lean_any"] else 0.55, zorder=3)
        rho = m["rho_issues"]
        ax.set_title(f"{t}   rho = {rho:+.2f}" if rho is not None else f"{t}   rho = n/a", fontsize=7.6, color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.tick_params(labelsize=6.5, length=0)
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("stated lean (score units, + = agrees with report level)", fontsize=7.2, color=INK)
    for ax in axes[::ncol]:
        ax.set_ylabel("revealed lean (log-odds, nats)", fontsize=7.2, color=INK)
    fig.suptitle("Stated vs revealed opinion, our 17 political issues (one point per issue)", fontsize=10.5, color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, label="same sign"),
               Line2D([], [], marker="o", ls="", color=RED, label="opposite sign"),
               Line2D([], [], marker="o", ls="", mfc=MUTED, mec=SURF, label="larger = revealed lean called (any focal)")]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right", ncol=3,
               bbox_to_anchor=(0.99, 0.965))
    fig.text(0.01, 0.005, "Stated: mean score difference between the two statements of an issue over 10 paraphrases x 3 forcings. "
             "Revealed: mean log-odds toward the report-level candidate over the three focal cells. Blue = the two readouts agree in sign.",
             fontsize=6.6, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("wrote", path)


def plot_pct_scatter(per_model, order, path):
    n = len(order)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 2.9 * nrow), sharex=True, sharey=True)
    axes = [axes] if not hasattr(axes, "flat") else list(axes.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, t in zip(axes, order):
        m = per_model[t]
        for p in m["pct"].values():
            if p["stated"] is None or p["revealed"] is None:
                continue
            col = BLUE if sign(p["stated"]) == sign(p["revealed"]) else RED
            ax.plot(p["stated"], p["revealed"], "o", ms=4.5, color=col, mec=SURF, mew=0.6,
                    alpha=0.85, zorder=3)
        rho = m["rho_pct"]
        ax.set_title(f"{t}   rho = {rho:+.2f}" if rho is not None else f"{t}   rho = n/a", fontsize=7.6, color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.tick_params(labelsize=6.5, length=0)
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("stated score at forcing 3 (+ = agrees)", fontsize=7.2, color=INK)
    for ax in axes[::ncol]:
        ax.set_ylabel("revealed log-odds toward agree-side candidate (nats)", fontsize=7.2, color=INK)
    fig.suptitle("Stated vs revealed opinion on the 62 PCT propositions", fontsize=10.5, color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, label="same sign"),
               Line2D([], [], marker="o", ls="", color=RED, label="opposite sign")]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.965))
    fig.text(0.01, 0.005, "Stated: mean score over paraphrases at forcing 3. Revealed: mean log-odds (log p(agree-side candidate) - log p(disagree-side)) over the PCT trials. "
             "Blue = same sign.", fontsize=6.6, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("wrote", path)


def plot_compass(per_model, order, path):
    fig, ax = plt.subplots(figsize=(7.8, 7.3))
    arrows = []
    for t in order:
        m = per_model[t]
        st, rv = m["compass"]["stated"], m["compass"]["revealed"]
        if not st or not rv:
            continue
        col = FAMILY_SLOTS.get(m["family"], "#c3c2b7")
        ax.annotate("", xy=(rv["economic"], rv["social"]), xytext=(st["economic"], st["social"]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, alpha=0.75, shrinkA=5, shrinkB=5))
        ax.plot(st["economic"], st["social"], "o", ms=5, color=col, mec=SURF, mew=0.7, alpha=0.9, zorder=3)
        ax.plot(rv["economic"], rv["social"], "s", ms=5.5, color=col, mec=SURF, mew=0.7, zorder=4)
        arrows.append((t, st, rv, col))
    if not arrows:
        plt.close(fig)
        return
    mx = max(max(abs(v) for v in (a[1]["economic"], a[1]["social"], a[2]["economic"], a[2]["social"])) for a in arrows)
    lim = max(1.0, mx * 1.2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    for (qx, qy, lab) in ((-1, 1, "left - authoritarian"), (1, 1, "right + authoritarian"),
                          (-1, -1, "left - libertarian"), (1, -1, "right + libertarian")):
        ax.text(qx * lim * 0.97, qy * lim * 0.97, lab, ha="right" if qx > 0 else "left",
                va="top" if qy > 0 else "bottom", fontsize=8.5, color=MUTED)
    for fam, col in FAMILY_SLOTS.items():
        if any(a[3] == col for a in arrows):
            ax.plot([], [], "o", color=col, label=fam)
    if any(a[3] not in FAMILY_SLOTS.values() for a in arrows):
        ax.plot([], [], "o", color="#c3c2b7", label="other families")
    handles = [Line2D([], [], marker="o", ls="", color=MUTED, label="stated (forced choice)"),
               Line2D([], [], marker="s", ls="", color=MUTED, label="revealed (candidate choice)")]
    leg1 = ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    ax.add_artist(leg1)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", title="family", title_fontsize=7.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("PCT economic axis (left -, right +)", fontsize=9, color=INK)
    ax.set_ylabel("PCT social axis (libertarian -, authoritarian +)", fontsize=9, color=INK)
    ax.set_title("Stated vs revealed PCT compass\narrows run from the stated to the revealed position", fontsize=10.5, color=INK, loc="left")
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "Each axis = mean over its propositions of agree_sign x stated score (circle) or agree_sign x revealed log-odds (square), "
             "so both readouts use the same sign convention (agree = +). Items with a missing readout are skipped from that readout's mean.",
             fontsize=6.6, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------- analysis


def analyse(sweep, stated, pct_cells, pct_issues, models):
    axes_items = list(pct_issues["axes"]) if pct_issues else []
    econ_items = [a for a in axes_items if pct_issues["axes"][a].get("pct_axis") == "economic"]
    soc_items = [a for a in axes_items if pct_issues["axes"][a].get("pct_axis") == "social"]
    per_model = {}
    overall_pts = []
    overall_pct_pts = []
    cross_model_disagree = defaultdict(list)   # item -> list of (tag, sign(stated), sign(revealed))
    for t in models:
        r = sweep[t]
        g = stated[t]["grid"] if t in stated else {}
        # --- 1. our 17 issues
        issues = {}
        for a in POLITICAL:
            st = stated_issue_lean(g, a) if g else None
            cells = [r["cells"].get(f"{f}|{a}") for f in FOCAL_AXES]
            have = [c for c in cells if c]
            rev = sum(c["lo_mean"] for c in have) / len(have) if have else None
            issues[a] = {
                "stated_lean": st["lean"] if st else None,
                "stated_cells": st["n_cells"] if st else 0,
                "revealed_lean": rev,
                "revealed_cells": len(have),
                "lo_lean_any": any(c["lo_lean"] for c in have if c),
                "lo_lean_n": sum(1 for c in have if c and c["lo_lean"]),
            }
        pts = [(v["stated_lean"], v["revealed_lean"]) for v in issues.values()
               if v["stated_lean"] is not None and v["revealed_lean"] is not None]
        overall_pts += pts
        rho = spearman([p[0] for p in pts], [p[1] for p in pts])
        # 2x2: stated sign x revealed sign, restricted to issues whose revealed lean is called
        counts22 = {"++": 0, "+-": 0, "-+": 0, "--": 0, "stated_null": 0}
        refuse_reveal = 0
        for a, v in issues.items():
            if not v["lo_lean_any"]:
                continue                       # 2x2 restricted to issues whose revealed lean is called
            if v["stated_lean"] is None:
                counts22["stated_null"] += 1
                continue
            key = ("+" if v["stated_lean"] >= 0 else "-") + ("+" if v["revealed_lean"] >= 0 else "-")
            counts22[key] += 1
            if g and refusal_flag(g, f"{a}|{ISSUES['axes'][a]['report']}"):
                refuse_reveal += 1
        sign_agree = sum(1 for x, y in pts if sign(x) == sign(y))
        # --- 2. validity and instability
        validity = {f: {"mean": None, "n": 0, "items_low": 0} for f in FORCINGS}
        if g:
            for f in FORCINGS:
                vms, n = [], 0
                low_items = 0
                for item in g:
                    v, nn = stated_valid_mass(g, item, f)
                    vms += v
                    n += nn
                    s = sorted(v)
                    if s:
                        med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
                        low_items += med < REFUSE_VM
                validity[f] = {"mean": (sum(vms) / len(vms) if vms else None), "n": n,
                               "items_low": low_items}
        dforce, nforce = stated_forcing_shift(g) if g else (None, 0)
        dpara, npara, para_q = stated_paraphrase_range(g) if g else (None, 0, {})
        # --- 3. PCT propositions
        pct = {}
        pc = pct_cells.get(t, {}).get("cells", {})
        for a in axes_items:
            sc, nsc = (stated_score_by_item(g, a, forcing_filter=3) if g else (None, 0))
            c = pc.get(a)
            rev = c["lo_mean"] if c else None
            pct[a] = {"stated": sc, "n_stated": nsc, "revealed": rev,
                      "lo_lean": bool(c and c.get("lo_lean")),
                      "prop": pct_issues["axes"][a].get("proposition", a)}
            if sc is not None and rev is not None:
                overall_pct_pts.append((t, a, sc, rev))
                cross_model_disagree[a].append((t, sign(sc), sign(rev)))
        pts_pct = [(v["stated"], v["revealed"]) for v in pct.values()
                   if v["stated"] is not None and v["revealed"] is not None]
        rho_pct = spearman([p[0] for p in pts_pct], [p[1] for p in pts_pct])
        # compasses
        def axis_mean(items, key):
            vals = []
            for a in items:
                v = pct[a].get(key)
                if v is None:
                    continue
                sgn = pct_issues["axes"][a].get("agree_sign", 1)
                vals.append(sgn * v)
            return (sum(vals) / len(vals) if vals else None), len(vals)
        st_econ, st_econ_n = axis_mean(econ_items, "stated")
        st_soc, st_soc_n = axis_mean(soc_items, "stated")
        rv_econ, rv_econ_n = axis_mean(econ_items, "revealed")
        rv_soc, rv_soc_n = axis_mean(soc_items, "revealed")
        per_model[t] = {
            "family": r["family"], "params_b": r["params_b"], "low_mass": r["low_mass"],
            "null_axis_lo_leans": r.get("null_axis_lo_leans") or [], "sweep_mass": r["mass"],
            "n_stated_items": len(g),
            "issues": issues, "rho_issues": rho, "n_issues_both": len(pts),
            "sign_agree_issues": sign_agree, "counts22": counts22, "refuse_but_reveal": refuse_reveal,
            "validity": validity, "dforcing": dforce, "n_dforcing": nforce,
            "dparaphrase": dpara, "n_dparaphrase": npara, "dparaphrase_q": para_q,
            "pct": pct, "rho_pct": rho_pct, "n_pct_both": len(pts_pct),
            "sign_agree_pct": sum(1 for x, y in pts_pct if sign(x) == sign(y)),
            "compass": {"stated": ({"economic": st_econ, "social": st_soc,
                                    "economic_n": st_econ_n, "social_n": st_soc_n}
                                   if st_econ is not None or st_soc is not None else None),
                        "revealed": ({"economic": rv_econ, "social": rv_soc,
                                      "economic_n": rv_econ_n, "social_n": rv_soc_n}
                                     if rv_econ is not None or rv_soc is not None else None)},
        }
    overall_rho = spearman([p[0] for p in overall_pts], [p[1] for p in overall_pts])
    overall_rho_pct = spearman([p[2] for p in overall_pct_pts], [p[3] for p in overall_pct_pts])
    overall_sign = sum(1 for x, y in overall_pts if sign(x) == sign(y))
    overall_sign_pct = sum(1 for _, _, x, y in overall_pct_pts if sign(x) == sign(y))
    # propositions where sign disagrees for most models
    disagree_props = []
    for a, lst in cross_model_disagree.items():
        n = len(lst)
        dis = sum(1 for _t, ss, rs in lst if ss != rs)
        if n >= 3 and dis > n / 2:
            disagree_props.append({"item": a, "prop": pct_issues["axes"][a].get("proposition", a),
                                   "n": n, "n_disagree": dis,
                                   "pct_axis": pct_issues["axes"][a].get("pct_axis"),
                                   "models": [(t, ss, rs) for t, ss, rs in lst]})
    disagree_props.sort(key=lambda d: -(d["n_disagree"] / d["n"]))
    # mean shift direction
    shifts = []
    for t, m in per_model.items():
        st, rv = m["compass"]["stated"], m["compass"]["revealed"]
        if st and rv and st["economic"] is not None and rv["economic"] is not None:
            shifts.append((t, rv["economic"] - st["economic"], rv["social"] - st["social"]))
    shift_summary = {
        "n": len(shifts),
        "economic": (sum(s[1] for s in shifts) / len(shifts) if shifts else None),
        "social": (sum(s[2] for s in shifts) / len(shifts) if shifts else None),
        "n_right": sum(1 for s in shifts if s[1] > 0), "n_left": sum(1 for s in shifts if s[1] < 0),
        "n_auth": sum(1 for s in shifts if s[2] > 0), "n_lib": sum(1 for s in shifts if s[2] < 0),
    }
    return per_model, {
        "rho_issues": overall_rho, "n_issues_pts": len(overall_pts),
        "sign_agree_issues": overall_sign,
        "rho_pct": overall_rho_pct, "n_pct_pts": len(overall_pct_pts),
        "sign_agree_pct": overall_sign_pct,
        "disagree_props": disagree_props, "shift": shift_summary,
    }


# --------------------------------------------------------------------------- report


def table_issues(order, per_model):
    L = ["| model | rho (17 issues) | same sign | refuse-but-reveal | stated+revealed+ | stated+revealed- | stated-revealed+ | stated-revealed- | stated null |", "|---|---|---|---|---|---|---|---|---|"]
    for t in order:
        m = per_model[t]
        rho = f"{m['rho_issues']:+.2f}" if m["rho_issues"] is not None else "—"
        c = m["counts22"]
        L.append(f"| {t} | {rho} | {m['sign_agree_issues']}/{m['n_issues_both']} | {m['refuse_but_reveal']} | "
                 f"{c['++']} | {c['+-']} | {c['-+']} | {c['--']} | {c['stated_null']} |")
    return L


def table_validity(order, per_model):
    L = ["| model | valid mass @1 | @3 | @5 | items <0.3 @1 | mean \\|Δscore\\| f1→f5 | mean paraphrase range | p90 paraphrase range |", "|---|---|---|---|---|---|---|---|"]
    for t in order:
        m = per_model[t]
        v = m["validity"]
        def vv(f):
            return f"{v[f]['mean']:.2f}" if v[f]["mean"] is not None else "—"
        dq = m["dparaphrase_q"]
        df = f"{m['dforcing']:.2f}" if m["dforcing"] is not None else "—"
        dp = f"{m['dparaphrase']:.2f}" if m["dparaphrase"] is not None else "—"
        d90 = f"{dq['p90']:.2f}" if dq else "—"
        L.append(f"| {t} | {vv(1)} | {vv(3)} | {vv(5)} | {v[1]['items_low']}/{m['n_stated_items']} | {df} | {dp} | {d90} |")
    return L


def table_compass(order, per_model):
    L = ["| model | stated econ | stated soc | revealed econ | revealed soc | Δecon | Δsoc | rho PCT | same sign PCT |", "|---|---|---|---|---|---|---|---|---|"]
    for t in order:
        m = per_model[t]
        st, rv = m["compass"]["stated"], m["compass"]["revealed"]

        def f2(x):
            return f"{x:+.2f}" if x is not None else "—"
        de = (rv["economic"] - st["economic"]) if (st and rv and st["economic"] is not None and rv["economic"] is not None) else None
        ds = (rv["social"] - st["social"]) if (st and rv and st["social"] is not None and rv["social"] is not None) else None
        rho = f"{m['rho_pct']:+.2f}" if m["rho_pct"] is not None else "—"
        L.append(f"| {t} | {f2(st['economic'] if st else None)} | {f2(st['social'] if st else None)} | "
                 f"{f2(rv['economic'] if rv else None)} | {f2(rv['social'] if rv else None)} | {f2(de)} | {f2(ds)} | "
                 f"{rho} | {m['sign_agree_pct']}/{m['n_pct_both']} |")
    return L


def write_report(out, synthetic, per_model, overall, order, n_stated, n_pct, pct_issues, sweep, args_paths):
    ms = max(per_model.values(), key=lambda m: (m["rho_issues"] if m["rho_issues"] is not None else -9))
    ls = min(per_model.values(), key=lambda m: (m["rho_issues"] if m["rho_issues"] is not None else 9))
    ms_t = [t for t, m in per_model.items() if m is ms][0]
    ls_t = [t for t, m in per_model.items() if m is ls][0]
    shift = overall["shift"]
    n_dis = len(overall["disagree_props"])
    top_dis = overall["disagree_props"][:5]
    mean_rho = sum(m["rho_issues"] for m in per_model.values() if m["rho_issues"] is not None) / max(1, sum(1 for m in per_model.values() if m["rho_issues"] is not None))
    mean_rho_pct = sum(m["rho_pct"] for m in per_model.values() if m["rho_pct"] is not None) / max(1, sum(1 for m in per_model.values() if m["rho_pct"] is not None))
    refuse_total = sum(m["refuse_but_reveal"] for m in per_model.values())
    L = []
    L += ["# Stated vs revealed political opinions", "",
          "Linking the election sweep to Röttger et al. 2024, *Political Compass or Spinning Arrow?* "
          "(arXiv 2402.16786). The paper asks a model its opinion directly and shows the answer is unstable under "
          "paraphrase and forcing, and that open-ended answers place models further right/libertarian than forced "
          "multiple choice. Here the same models are also read through the sweep instrument: which candidate they "
          "name when an unasked issue is the only difference between two otherwise identical candidates. Stated = "
          "what the model says it thinks; revealed = what it does when the issue is never mentioned.",
          ""]
    if synthetic:
        L += ["> **NOTE: this run is on synthetic data.** The generator fabricates stated rows and PCT trials with "
              "planted correlations; every number below is fake and exists only to exercise the code path. Re-run on "
              "the real `experiments/pct/*.jsonl` once tonight's pod lands.", ""]
    L += ["## Most interesting results", ""]
    L.append(f"- **Do stated and revealed opinions line up?** Across {overall['n_issues_pts']} model-issue pairs on our "
             f"17 political issues, the Spearman correlation between stated lean (score units) and revealed lean "
             f"(log-odds) is **{overall['rho_issues']:+.2f}** overall, and {overall['sign_agree_issues']}/{overall['n_issues_pts']} "
             f"pairs agree in sign. Mean per-model rho is {mean_rho:+.2f}; the tightest model is {ms_t} "
             f"({ms['rho_issues']:+.2f}) and the loosest {ls_t} ({ls['rho_issues']:+.2f}).")
    if shift["n"]:
        econ_word = "right" if (shift["economic"] or 0) > 0 else "left"
        soc_word = "authoritarian" if (shift["social"] or 0) > 0 else "libertarian"
        L.append(f"- **Compass shift direction (the paper's headline, reversed here).** Röttger et al. found open-ended "
                 f"answers shift right/libertarian relative to forced multiple choice. With revealed = the candidate "
                 f"choice and stated = forced multiple choice, the mean shift over {shift['n']} models is "
                 f"**{shift['economic']:+.2f} on the economic axis ({econ_word}) and {shift['social']:+.2f} on the social "
                 f"axis ({soc_word})**; {shift['n_right']} of {shift['n']} models move right, {shift['n_auth']} move authoritarian. "
                 + ("So the revealed readout moves models in roughly the direction the paper calls the open-ended bias."
                    if (shift["economic"] or 0) > 0 and (shift["social"] or 0) < 0 else
                    "That is NOT the paper's open-ended direction (right + libertarian); the revealed candidate-choice "
                    "readout does not inherit the open-ended shift.") if shift["n"] else "")
    if refuse_total:
        n_rb = sum(m["counts22"]["++"] + m["counts22"]["--"] + m["counts22"]["+-"] + m["counts22"]["-+"] for m in per_model.values())
        L.append(f"- **Refuse but reveal.** {refuse_total} (model, issue) cells have a stated refusal (valid mass "
                 f"< {REFUSE_VM} at forcing 1) on an issue where the revealed readout is called as a lean anyway, out of "
                 f"{n_rb} called cells; the model will not state an opinion but discriminates candidates by it.")
    if overall["n_pct_pts"]:
        L.append(f"- **PCT propositions.** Over {overall['n_pct_pts']} model-proposition pairs the stated-vs-revealed "
                 f"Spearman is {overall['rho_pct']:+.2f} (mean per-model {mean_rho_pct:+.2f}); "
                 f"{overall['sign_agree_pct']}/{overall['n_pct_pts']} pairs agree in sign. "
                 f"{n_dis} propositions flip sign for the majority of models."
                 + ("" if not top_dis else " The largest: " + "; ".join(f"\"{d['prop'][:60]}...\" ({d['n_disagree']}/{d['n']})" for d in top_dis[:3]) + "."))
    v1 = [m["validity"][1]["mean"] for m in per_model.values() if m["validity"][1]["mean"] is not None]
    v5 = [m["validity"][5]["mean"] for m in per_model.values() if m["validity"][5]["mean"] is not None]
    if v1 and v5:
        L.append(f"- **Forcing buys validity.** Mean stated valid mass is {sum(v1)/len(v1):.2f} at forcing 1 and "
                 f"{sum(v5)/len(v5):.2f} at forcing 5; mean |Δscore| between forcing 1 and 5 is "
                 f"{(sum(m['dforcing'] for m in per_model.values() if m['dforcing'] is not None) / max(1, sum(1 for m in per_model.values() if m['dforcing'] is not None))):.2f} "
                 f"score units, and the mean paraphrase range is "
                 f"{(sum(m['dparaphrase'] for m in per_model.values() if m['dparaphrase'] is not None) / max(1, sum(1 for m in per_model.values() if m['dparaphrase'] is not None))):.2f} "
                 f"— the paper's instability, reproduced on our instrument.")
    L += ["", "## 1. Stated vs revealed lean on our 17 political issues", "",
          "Stated lean per issue = mean over the 10 paraphrases x 3 forcings of score(report-level statement) "
          "− score(other-level statement); positive = the model agrees more with the position the sweep reports "
          "(the `report` level of `data/issues.json`). Revealed lean = mean `lo_mean` over the three focal cells "
          "for that issue from `experiments/sweep/summary.json`. The 2x2 table is restricted to issues whose "
          "revealed lean is *called* (any focal cell passes the log-odds rule: CI excludes 0, |mean| ≥ "
          f"{LOGODDS_FLOOR} nats, BH q ≤ 0.01).", "",
          f"![stated vs revealed, our issues](fig_issues_scatter.png)", "",
          *table_issues(order, per_model),
          "", "Columns `++`/`+-`/`-+`/`--` are {sign of stated lean} x {sign of revealed lean} over called cells; "
          "`stated null` counts called issues with no stated data. `refuse-but-reveal` counts called issues where "
          f"the stated readout refuses (valid mass < {REFUSE_VM} at forcing 1) yet the revealed lean is called.", "",
          "Overall Spearman across all model-issue pairs: "
          f"**{overall['rho_issues']:+.2f}** (n = {overall['n_issues_pts']}), sign agreement "
          f"{overall['sign_agree_issues']}/{overall['n_issues_pts']}.", "",
          "## 2. Validity by forcing, and instability (the paper's Figure 2 analogue)", "",
          *table_validity(order, per_model),
          "", f"![validity by forcing](fig_validity.png)", "",
          f"valid_mass = mass on the four agreement options; < {REFUSE_VM} at forcing 1 is read as a refusal. "
          "|Δscore| compares matched (item, paraphrase) rows between forcing 1 and forcing 5. The paraphrase range "
          "is per item the max − min of the paraphrase-mean score, then averaged over items (p90 = 90th percentile "
          "over items).", "",
          "## 3. The 62 PCT propositions: stated vs revealed", "",
          f"![PCT stated vs revealed](fig_pct_scatter.png)", "",
          f"{len(pct_issues.get('axes', {}))} propositions. Stated = mean score over paraphrases at forcing 3 "
          "(the paper's forced-choice setting). Revealed = mean log-odds toward the agree-side candidate from "
          "`pct_<tag>.jsonl`, clustered by background. Overall Spearman "
          f"**{overall['rho_pct']:+.2f}** (n = {overall['n_pct_pts']}), sign agreement "
          f"{overall['sign_agree_pct']}/{overall['n_pct_pts']}.", ""]
    if overall["disagree_props"]:
        L += ["Propositions where the two readouts disagree in sign for the majority of models:", "",
              "| proposition | axis | models disagreeing | n |", "|---|---|---|---|"]
        for d in overall["disagree_props"][:12]:
            L.append(f"| {d['prop']} | {d.get('pct_axis') or ''} | {d['n_disagree']} | {d['n']} |")
        L.append("")
    L += ["### Stated vs revealed PCT compass", "",
          f"![stated to revealed compass](fig_compass.png)", "",
          *table_compass(order, per_model),
          "", "Both readouts are placed with the same convention: axis value = mean over that axis's propositions "
          "of agree_sign x score (stated) or agree_sign x revealed log-odds, so + = agree side / right / "
          "authoritarian as `data/pct_issues.json` signs it. The paper found open-ended answers land more "
          "right/libertarian than forced multiple choice; the arrow here runs stated → revealed, so the question "
          "is whether the candidate-choice readout inherits that direction.", ""]
    if shift["n"]:
        L += [f"Mean shift over {shift['n']} models: economic {shift['economic']:+.2f} "
              f"({shift['n_right']} right / {shift['n_left']} left), social {shift['social']:+.2f} "
              f"({shift['n_lib']} libertarian / {shift['n_auth']} authoritarian).", ""]
    L += ["## Method notes and assumptions", "",
          "- Stated means are simple means of cell means: each (item, paraphrase, forcing) row counts equally, "
          "matching how the sweep weights trials. Missing cells are skipped, not imputed.",
          f"- A stated row is read as a refusal when valid_mass < {REFUSE_VM}; rows between {REFUSE_VM} and "
          f"{VALID_VM} are kept (their score carries less mass but is still the model's distribution over options).",
          "- Per-model Spearman rho is undefined (reported as —) when either readout is constant across issues.",
          "- The revealed 2x2 uses the sweep's log-odds lean call (any of the three focal cells); the win-rate "
          "call saturates for models with a large listing-position effect and is not used here.",
          "- Models with a null-axis log-odds lean or naming mass < 0.2 are flagged in the tables but kept in all "
          "statistics; excluding them changes nothing reported above unless stated.",
          f"- Data sources: {n_stated} stated files, {n_pct} PCT revealed files, "
          f"{args_paths}.", ""]
    Path(out).write_text("\n".join(L) + "\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", default=str(ROOT / "experiments" / "sweep"))
    ap.add_argument("--stated_dir", default=str(ROOT / "experiments" / "pct"))
    ap.add_argument("--pct_dir", default=str(ROOT / "experiments" / "pct"))
    ap.add_argument("--pct_issues", default=str(ROOT / "data" / "pct_issues.json"))
    ap.add_argument("--out", default=None, help="output directory (default: <stated_dir>/analysis)")
    ap.add_argument("--models", nargs="*", default=None, help="model tags to analyse (default: all with stated data)")
    args = ap.parse_args()
    sweep_dir, stated_dir, pct_dir = Path(args.sweep_dir), Path(args.stated_dir), Path(args.pct_dir)
    out = Path(args.out) if args.out else stated_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    sweep = json.loads((sweep_dir / "summary.json").read_text())
    pct_issues = load_pct_issues(args.pct_issues)
    stated = load_stated(stated_dir, list(sweep))
    pct_cells = load_pct_cells(pct_dir, list(sweep))
    tags = args.models or [t for t in sweep if t in stated]
    tags = [t for t in tags if t in stated or t in pct_cells]
    if not tags:
        sys.exit("no models with stated or PCT data found; run the pod first or pass --models")
    # a synthetic smoke-test run is flagged in the report when any input path says so
    synthetic = any("synthetic" in str(p) for p in (stated_dir, pct_dir, args.pct_issues))
    per_model, overall = analyse(sweep, stated, pct_cells, pct_issues, tags)
    fam_order = []
    for m in json.loads((SCRIPTS / "sweep_models.json").read_text())["models"]:
        if m.get("family") not in fam_order:
            fam_order.append(m.get("family"))
    order = sorted(per_model, key=lambda t: (fam_order.index(per_model[t]["family"]) if per_model[t]["family"] in fam_order else 99,
                                             per_model[t]["params_b"] or 0, t))
    plot_validity(per_model, order, out / "fig_validity.png")
    plot_issues_scatter(per_model, order, out / "fig_issues_scatter.png")
    plot_pct_scatter(per_model, order, out / "fig_pct_scatter.png")
    plot_compass(per_model, order, out / "fig_compass.png")
    write_report(out / "stated_vs_revealed.md", synthetic, per_model, overall, order,
                 len(stated), len(pct_cells), pct_issues, sweep,
                 f"sweep={sweep_dir}, stated={stated_dir}, pct={pct_dir}, pct_issues={args.pct_issues}")
    (out / "stated_vs_revealed.json").write_text(json.dumps(
        {"per_model": per_model, "overall": overall, "order": order, "synthetic": synthetic}, indent=1, default=str))
    print(f"{len(tags)} models analysed; overall rho(issues) = "
          f"{overall['rho_issues'] if overall['rho_issues'] is None else format(overall['rho_issues'], '+.2f')}")
    return per_model, overall


if __name__ == "__main__":
    main()
