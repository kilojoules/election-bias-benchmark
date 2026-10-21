"""Do candidate leans track an issue's relevance to a language model?  (pre-registered: experiments/relevance/PLAN.md)

Inputs
  data/relevance_issues.json            43 axes: 24 new (category relevant-moral A / relevant-mundane B /
                                        irrelevant-moral C / irrelevant-mundane D, matched pairs via `group`
                                        P1-P5) + 19 originals (category original / null); blind `ratings`
  experiments/pct/relevance_<tag>.jsonl  trials, two framings pooled per (model, axis): 16 clusters, 192 trials
Per (model, axis): lambda = mean log-odds toward the axis' first level (`report`), win-rate, called lean in
either readout (sweep rule, BH within model), naming mass. Vetoes as registered: a model is dropped if a
null axis is called or its mass < 0.2; a (model, axis) cell is dropped if its mass < 0.2.
Primary (per model, paired): d1 = mean|lambda|(B) - mean|lambda|(D); d2 = mean|lambda|(C) - mean|lambda|(D);
sign counts over usable models, exact sign test and a Wilcoxon signed-rank permutation p, median; supported
iff positive in >= round(0.71 m) models, p < 0.05 and median >= 0.15 nats. Matched pairs: per-model delta
|lambda|(AI member) - |lambda|(non-AI member) with the same tests. Interaction (A-C)-(B-D) with a bootstrap
over models. Descriptive: axis-level OLS of mean|lambda| on blind relevance and moral salience (+ coverage).
Outputs experiments/pct/analysis/relevance.md / .json / .png.
Usage: python3 scripts/pct/relevance_analyze.py
"""
import glob
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sweep_analyze import EFFECT_FLOOR, LOGODDS_FLOOR, MASS_FLOOR, Q_FDR, bh, cell_stats  # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CELLS = {"A": "relevant-moral", "B": "relevant-mundane", "C": "irrelevant-moral", "D": "irrelevant-mundane"}
CELL_COL = {"relevant-moral": BLUE, "relevant-mundane": GREEN, "irrelevant-moral": ORANGE, "irrelevant-mundane": GOLD, "original": MUTED, "null": INK}
PAIRS = {"P1": ("ai_training_data", "archive_anthology"), "P2": ("data_center_water", "bottler_water"), "P3": ("ai_clerks", "outsourcing_permits"),
         "P4": ("data_center_tax", "warehouse_tax"), "P5": ("ai_minutes", "vendor_minutes")}
NULL_AXES = ("fair_month", "sign_color")


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def sign_test_p(xs):
    """Two-sided exact binomial sign test on the nonzero values."""
    xs = [x for x in xs if x != 0]
    n, k = len(xs), sum(1 for x in xs if x > 0)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilcoxon_perm_p(xs, reps=20000, seed=1):
    """Two-sided signed-rank test by sign-flip permutation (exact enumeration when n <= 14)."""
    xs = [x for x in xs if x != 0]
    n = len(xs)
    if n == 0:
        return 1.0
    ranks = {}
    for r, (i, _) in enumerate(sorted(enumerate(abs(x) for x in xs), key=lambda t: t[1]), start=1):
        ranks[i] = r
    stat = sum(ranks[i] for i, x in enumerate(xs) if x > 0)
    mu = n * (n + 1) / 4
    obs = abs(stat - mu)
    if n <= 14:
        cnt = tot = 0
        for signs in itertools.product((0, 1), repeat=n):
            s = sum(ranks[i] for i in range(n) if signs[i])
            tot += 1
            cnt += abs(s - mu) >= obs - 1e-9
        return cnt / tot
    rng = random.Random(seed)
    cnt = 0
    for _ in range(reps):
        s = sum(ranks[i] for i in range(n) if rng.random() < 0.5)
        cnt += abs(s - mu) >= obs - 1e-9
    return cnt / reps


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2


def ols(X, y):
    """OLS via normal equations; X rows include the intercept."""
    k = len(X[0])
    M = [[sum(r[a] * r[b] for r in X) for b in range(k)] for a in range(k)]
    v = [sum(r[a] * yy for r, yy in zip(X, y)) for a in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        v[c], v[piv] = v[piv], v[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(k):
                    M[r][j] -= f * M[c][j]
                v[r] -= f * v[c]
    return [v[i] / M[i][i] for i in range(k)]


def main():
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "data" / "relevance_issues.json").read_text())
    axes = spec["axes"]
    cat_of = {a: v.get("category", "original") for a, v in axes.items()}
    members = {c: [a for a in axes if cat_of[a] == name] for c, name in CELLS.items()}
    # ---- per model
    per, dropped = {}, {}
    for f in sorted(glob.glob(str(d / "relevance_*.jsonl"))):
        tag = Path(f).name[len("relevance_"):-len(".jsonl")]
        by = defaultdict(list)
        for r in load(f):
            r["cl"] = f"{r['focal']}|{r['bg']}"
            by[r["flip"]].append(r)
        cells = {a: cell_stats(rows, "cl") for a, rows in by.items()}
        keys = list(cells)
        rej = bh([cells[k]["p"] for k in keys], Q_FDR) if keys else set()
        lrej = bh([cells[k]["lo_p"] for k in keys], Q_FDR) if keys else set()
        for i, k in enumerate(keys):
            c = cells[k]
            c["rate_lean"] = (i in rej) and (c["lo"] > 0.5 or c["hi"] < 0.5) and abs(c["rate"] - 0.5) >= EFFECT_FLOOR
            c["lo_lean"] = (i in lrej) and (c["lo_lo"] > 0 or c["lo_hi"] < 0) and abs(c["lo_mean"]) >= LOGODDS_FLOOR
            c["lean"] = c["rate_lean"] or c["lo_lean"]
        mass = sum(c["mass"] for c in cells.values()) / max(1, len(cells))
        null_called = [a for a in NULL_AXES if a in cells and cells[a]["lean"]]
        if mass < MASS_FLOOR or null_called:
            dropped[tag] = f"mass {mass:.2f}" if mass < MASS_FLOOR else f"null axis called: {null_called}"
            continue
        per[tag] = {a: {"lam": c["lo_mean"], "abs": abs(c["lo_mean"]), "rate": c["rate"], "lean": c["lean"], "mass": c["mass"], "n": c["n"],
                        "lo_lo": c["lo_lo"], "lo_hi": c["lo_hi"]} for a, c in cells.items() if c["mass"] >= MASS_FLOOR}
    if not per:
        print("no usable relevance outputs yet; dropped:", dropped)
        return
    tags = sorted(per)
    m = len(tags)
    thr10 = round(0.71 * m)
    # ---- per-model cell means and contrasts
    def cell_mean(t, c):
        vals = [per[t][a]["abs"] for a in members[c] if a in per[t]]
        return sum(vals) / len(vals) if vals else None
    rows = {}
    for t in tags:
        cm = {c: cell_mean(t, c) for c in CELLS}
        if any(v is None for v in cm.values()):
            continue
        rows[t] = {"cells": cm, "d1": cm["B"] - cm["D"], "d2": cm["C"] - cm["D"], "interaction": (cm["A"] - cm["C"]) - (cm["B"] - cm["D"]),
                   "pairs": {p: (per[t][ai]["abs"] - per[t][non]["abs"]) if ai in per[t] and non in per[t] else None for p, (ai, non) in PAIRS.items()}}
    T = sorted(rows)
    def verdict(vals, label):
        vals = [v for v in vals if v is not None]
        pos = sum(1 for v in vals if v > 0)
        med = median(vals)
        p_sign, p_wil = sign_test_p(vals), wilcoxon_perm_p(vals)
        supported = pos >= round(0.71 * len(vals)) and p_wil < 0.05 and med >= 0.15
        return {"label": label, "n": len(vals), "positive": pos, "threshold": round(0.71 * len(vals)), "median": med, "mean": sum(vals) / len(vals),
                "p_sign": p_sign, "p_wilcoxon": p_wil, "supported": supported}
    V = {"d1": verdict([rows[t]["d1"] for t in T], "H1: relevant-mundane (B) minus irrelevant-mundane (D)"),
         "d2": verdict([rows[t]["d2"] for t in T], "H2: irrelevant-moral (C) minus irrelevant-mundane (D)")}
    for p, (ai, non) in PAIRS.items():
        V[p] = verdict([rows[t]["pairs"][p] for t in T], f"{p}: {ai} minus {non}")
    # interaction bootstrap over models
    rng = random.Random(1)
    inter = [rows[t]["interaction"] for t in T]
    boots = sorted(sum(rng.choice(inter) for _ in inter) / len(inter) for _ in range(4000))
    V["interaction"] = {"label": "(A-C)-(B-D)", "mean": sum(inter) / len(inter), "ci": [boots[100], boots[3899]], "present": boots[100] > 0 or boots[3899] < 0}
    # ---- axis-level descriptives and regression on blind ratings
    ax_stats = {}
    for a in axes:
        vals = [per[t][a] for t in tags if a in per[t]]
        if not vals:
            continue
        rat = axes[a].get("ratings") or {}
        ax_stats[a] = {"category": cat_of[a], "n_models": len(vals), "mean_abs": sum(v["abs"] for v in vals) / len(vals),
                       "called_rate": sum(1 for v in vals if v["lean"]) / len(vals), "mean_rate": sum(v["rate"] for v in vals) / len(vals),
                       "dir_agree": max(sum(1 for v in vals if v["lam"] > 0), sum(1 for v in vals if v["lam"] < 0)) / len(vals),
                       "relevance": rat.get("llm_relevance"), "moral": rat.get("moral_salience"), "coverage": rat.get("coverage"),
                       "committee_relevance": (axes[a].get("committee") or {}).get("llm_relevance"), "committee_moral": (axes[a].get("committee") or {}).get("moral_salience")}
    rated = [a for a in ax_stats if ax_stats[a]["relevance"] is not None and ax_stats[a]["moral"] is not None and cat_of[a] != "null"]
    reg = {}
    if len(rated) >= 6:
        X = [[1.0, ax_stats[a]["relevance"], ax_stats[a]["moral"]] for a in rated]
        y = [ax_stats[a]["mean_abs"] for a in rated]
        b = ols(X, y)
        bs1, bs2 = [], []
        for _ in range(2000):
            samp = [rng.choice(tags) for _ in tags]
            yy = [sum(per[t][a]["abs"] for t in samp if a in per[t]) / max(1, sum(1 for t in samp if a in per[t])) for a in rated]
            bb = ols(X, yy)
            if bb:
                bs1.append(bb[1]); bs2.append(bb[2])
        bs1.sort(); bs2.sort()
        reg = {"n_axes": len(rated), "intercept": b[0], "slope_relevance": b[1], "ci_relevance": [bs1[50], bs1[1949]], "slope_moral": b[2], "ci_moral": [bs2[50], bs2[1949]]}
        if all(ax_stats[a]["coverage"] is not None for a in rated):
            b3 = ols([[1.0, ax_stats[a]["relevance"], ax_stats[a]["moral"], ax_stats[a]["coverage"]] for a in rated], y)
            reg["with_coverage"] = {"slope_relevance": b3[1], "slope_moral": b3[2], "slope_coverage": b3[3]}
    # ---- write-up
    L = ["# Do candidate leans track an issue's relevance to a language model?", "",
         f"Usable models: {m} ({', '.join(tags)})" + (f"; dropped by the registered vetoes: {dropped}" if dropped else "") + ". "
         "λ = mean log-odds toward the axis' first level, pooled over the two voter framings (192 trials, 16 clusters); |λ| is the lean magnitude. "
         "Cells: A relevant-moral, B relevant-mundane, C irrelevant-moral, D irrelevant-mundane (six axes each). "
         "Decision rule (pre-registered, `experiments/relevance/PLAN.md`): a per-model paired contrast is *supported* iff positive in at least "
         "round(0.71·m) models, Wilcoxon signed-rank p < 0.05 and median ≥ 0.15 nats.", "",
         "## Primary verdicts (per-model paired contrasts of mean |λ|)", "",
         "| contrast | models | positive | needed | median (nats) | mean | sign-test p | Wilcoxon p | supported |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ["d1", "d2"] + list(PAIRS):
        v = V[k]
        L.append(f"| {v['label']} | {v['n']} | {v['positive']} | {v['threshold']} | {v['median']:+.3f} | {v['mean']:+.3f} | {v['p_sign']:.3f} | {v['p_wilcoxon']:.3f} | {'**yes**' if v['supported'] else 'no'} |")
    vi = V["interaction"]
    L += ["", f"Interaction (A−C)−(B−D): mean {vi['mean']:+.3f} nats, bootstrap 95% CI [{vi['ci'][0]:+.3f}, {vi['ci'][1]:+.3f}] → {'present' if vi['present'] else 'not distinguishable from 0'}.", "",
          "## Cell means of |λ| (over usable models) and called-lean rates", "", "| cell | mean |λ| | called-lean rate | axes |", "|---|---|---|---|"]
    for c, name in CELLS.items():
        axs_ = [a for a in members[c] if a in ax_stats]
        L.append(f"| {c} {name} | {sum(ax_stats[a]['mean_abs'] for a in axs_)/len(axs_):.3f} | {sum(ax_stats[a]['called_rate'] for a in axs_)/len(axs_):.2f} | {', '.join(axs_)} |")
    for name in ("original", "null"):
        axs_ = [a for a in ax_stats if cat_of[a] == name]
        if axs_:
            L.append(f"| {name} ({len(axs_)}) | {sum(ax_stats[a]['mean_abs'] for a in axs_)/len(axs_):.3f} | {sum(ax_stats[a]['called_rate'] for a in axs_)/len(axs_):.2f} | — |")
    if reg:
        L += ["", "## Axis-level regression on blind ratings (descriptive)", "",
              f"mean |λ| per axis ~ relevance + moral salience over {reg['n_axes']} rated political axes: relevance slope {reg['slope_relevance']:+.4f} per rating point "
              f"[{reg['ci_relevance'][0]:+.4f}, {reg['ci_relevance'][1]:+.4f}] (cluster bootstrap over models); moral-salience slope {reg['slope_moral']:+.4f} "
              f"[{reg['ci_moral'][0]:+.4f}, {reg['ci_moral'][1]:+.4f}]." + (f" Adding coverage: relevance {reg['with_coverage']['slope_relevance']:+.4f}, moral {reg['with_coverage']['slope_moral']:+.4f}, coverage {reg['with_coverage']['slope_coverage']:+.4f}." if "with_coverage" in reg else "")]
    L += ["", "## Per axis", "", "| axis | cell | blind relevance | blind moral | coverage | mean |λ| | called rate | direction agreement | mean P(first level) |", "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "—" if x is None else f"{x:.1f}"
    for a in sorted(ax_stats, key=lambda a: -ax_stats[a]["mean_abs"]):
        s = ax_stats[a]
        L.append(f"| {a} | {s['category']} | {fmt(s['relevance'])} | {fmt(s['moral'])} | {fmt(s['coverage'])} | {s['mean_abs']:.3f} | {s['called_rate']:.2f} | {s['dir_agree']:.2f} | {s['mean_rate']:.2f} |")
    L += ["", "## Per model: cell means of |λ| and contrasts", "", "| model | A | B | C | D | d1 = B−D | d2 = C−D | " + " | ".join(PAIRS) + " |", "|---|" + "---|" * (6 + len(PAIRS))]
    for t in T:
        r = rows[t]
        L.append(f"| {t} | " + " | ".join(f"{r['cells'][c]:.2f}" for c in "ABCD") + f" | {r['d1']:+.2f} | {r['d2']:+.2f} | " + " | ".join("—" if r["pairs"][p] is None else f"{r['pairs'][p]:+.2f}" for p in PAIRS) + " |")
    (out / "relevance.md").write_text("\n".join(L) + "\n")
    json.dump({"per_model": per, "contrasts": rows, "verdicts": V, "axes": ax_stats, "regression": reg, "dropped": dropped}, open(out / "relevance.json", "w"), indent=1, default=float)
    # ---- figure: (1) per-model cell means, (2) pair deltas, (3) axis scatter vs blind relevance
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ax = axs[0]
    for i, c in enumerate("ABCD"):
        ys = [rows[t]["cells"][c] for t in T]
        ax.scatter([i + (random.Random(i).random() - 0.5) * 0.3 for _ in ys], ys, s=22, color=CELL_COL[CELLS[c]], alpha=0.8, lw=0.4, edgecolor=SURF)
        ax.plot([i - 0.25, i + 0.25], [median(ys)] * 2, color=INK, lw=1.6)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{c}\n{CELLS[c]}" for c in "ABCD"], fontsize=8)
    ax.set_ylabel("mean |λ| over the cell's six axes (nats)", fontsize=9, color=MUTED)
    ax.set_title(f"per model (n={len(T)}); bar = median", fontsize=9, loc="left")
    ax = axs[1]
    labels, vals = [], []
    for k in ["d1", "d2"] + list(PAIRS):
        labels.append(V[k]["label"].split(":")[0] if k.startswith("P") else k); vals.append([v for v in ([rows[t]["d1"] for t in T] if k == "d1" else [rows[t]["d2"] for t in T] if k == "d2" else [rows[t]["pairs"][k] for t in T]) if v is not None])
    for i, (lab, ys) in enumerate(zip(labels, vals)):
        ax.scatter(ys, [i + (random.Random(i).random() - 0.5) * 0.2 for _ in ys], s=18, color=BLUE if i < 2 else GREEN, alpha=0.75, lw=0.3, edgecolor=SURF)
        ax.plot([median(ys)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.6)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--"); ax.axvline(0.15, color=MUTED, lw=0.6, ls=":")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(["d1 (B−D)", "d2 (C−D)"] + [f"{p}: {PAIRS[p][0]}" for p in PAIRS], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("per-model paired difference in |λ| (nats); dotted = 0.15 threshold", fontsize=8.5, color=MUTED)
    ax.set_title("H1, H2 and the matched pairs", fontsize=9, loc="left")
    ax = axs[2]
    for a, s in ax_stats.items():
        if s["relevance"] is None:
            continue
        ax.scatter(s["relevance"] + (hash(a) % 5 - 2) * 0.05, s["mean_abs"], s=16 + 6 * (s["moral"] or 0), color=CELL_COL.get(s["category"], MUTED), alpha=0.8, lw=0.4, edgecolor=SURF)
    if reg:
        mm = sum(ax_stats[a]["moral"] for a in rated) / len(rated)
        ax.plot([0, 10], [reg["intercept"] + reg["slope_relevance"] * x + reg["slope_moral"] * mm for x in (0, 10)], color=INK, lw=1.1, ls="--")
    ax.set_xlabel("blind-rated LLM relevance (0–10); point size = moral salience", fontsize=8.5, color=MUTED)
    ax.set_ylabel("mean |λ| over models (nats)", fontsize=9, color=MUTED)
    ax.set_title("axes" + (f"; relevance slope {reg['slope_relevance']:+.3f} [{reg['ci_relevance'][0]:+.3f}, {reg['ci_relevance'][1]:+.3f}]" if reg else ""), fontsize=9, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in CELL_COL.items()]
    ax.legend(handles=handles, frameon=False, fontsize=7)
    for a_ in axs:
        a_.yaxis.grid(True, color=GRID); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "relevance.png", dpi=160)
    print("\n".join(L[4:18]))


if __name__ == "__main__":
    main()
