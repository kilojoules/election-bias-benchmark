"""Do candidate leans track how relevant an issue is to a language model?

Inputs
  data/relevance_issues.json          the new axes (each with blind ratings: llm_relevance, moral_salience,
                                      balance, plus category) and the 19 original axes with the same ratings
  experiments/pct/relevance_<tag>.jsonl   this run's trials (new axes AND the originals, two framings)
Per (model, axis): pooled over framings, win-rate and log-odds lean toward the axis' `report` level, the
sweep's called-lean rule in either readout (BH within model). Per axis: mean |log-odds lean| over models,
fraction of models with a called lean. Tests (pre-registered in experiments/pct/analysis/RELEVANCE_PLAN.md):
  H1 relevance:  axis-level |lean| and called-rate increase with blind-rated LLM relevance, controlling for
                 moral salience (OLS on axis means; cluster-bootstrap over models for the CI of the slope)
  H2 salience:   the same for moral salience, controlling for relevance
  2x2 cell means (relevant/irrelevant x moral/mundane) with CIs; matched-pair contrasts.
Outputs experiments/pct/analysis/relevance.md / .json / .png.
Usage: python3 scripts/pct/relevance_analyze.py
"""
import glob
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
from sweep_analyze import EFFECT_FLOOR, LOGODDS_FLOOR, Q_FDR, bh, cell_stats  # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CELL_COL = {"relevant-moral": BLUE, "relevant-mundane": GREEN, "irrelevant-moral": ORANGE, "irrelevant-mundane": GOLD}


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def ols(xs, ys):
    """Two-predictor OLS y = a + b1*x1 + b2*x2 via normal equations (xs: list of (x1, x2))."""
    n = len(ys)
    X = [[1.0, x1, x2] for x1, x2 in xs]
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(3)] for a in range(3)]
    Xty = [sum(X[i][a] * ys[i] for i in range(n)) for a in range(3)]
    # solve 3x3
    import copy
    M = copy.deepcopy(XtX)
    v = Xty[:]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        v[c], v[piv] = v[piv], v[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(3):
                    M[r][k] -= f * M[c][k]
                v[r] -= f * v[c]
    return [v[i] / M[i][i] for i in range(3)]


def main():
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "data" / "relevance_issues.json").read_text())
    axes = spec["axes"]
    rated = {a: v for a, v in axes.items() if "ratings" in v}
    per = {}
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
        per[tag] = {a: {"rate": c["rate"], "lo_mean": c["lo_mean"], "lean": c["lean"], "n": c["n"], "mass": c["mass"]} for a, c in cells.items()}
    if not per:
        print("no relevance outputs yet")
        return
    tags = sorted(per)
    # axis-level aggregates over models
    ax_stats = {}
    for a in rated:
        vals = [per[t][a] for t in tags if a in per[t]]
        if not vals:
            continue
        ax_stats[a] = {"n_models": len(vals), "mean_abs_lo": sum(abs(v["lo_mean"]) for v in vals) / len(vals),
                       "called_rate": sum(1 for v in vals if v["lean"]) / len(vals),
                       "mean_signed_rate": sum(v["rate"] for v in vals) / len(vals),
                       "same_dir": max(sum(1 for v in vals if v["lo_mean"] > 0), sum(1 for v in vals if v["lo_mean"] < 0)) / len(vals),
                       **{k: rated[a]["ratings"][k] for k in ("llm_relevance", "moral_salience")}, "category": rated[a].get("category", "original")}
    A = sorted(ax_stats)
    # regressions on axis means; cluster bootstrap over models for slope CIs
    def fit(stat_key, subset):
        xs = [(ax_stats[a]["llm_relevance"], ax_stats[a]["moral_salience"]) for a in subset]
        ys = [ax_stats[a][stat_key] for a in subset]
        return ols(xs, ys)
    def boot_slopes(stat_key, subset, reps=2000, seed=1):
        rng = random.Random(seed)
        b1s, b2s = [], []
        for _ in range(reps):
            sample = [rng.choice(tags) for _ in tags]
            ys = []
            for a in subset:
                vals = [per[t][a] for t in sample if a in per[t]]
                if stat_key == "mean_abs_lo":
                    ys.append(sum(abs(v["lo_mean"]) for v in vals) / max(1, len(vals)))
                else:
                    ys.append(sum(1 for v in vals if v["lean"]) / max(1, len(vals)))
            b = ols([(ax_stats[a]["llm_relevance"], ax_stats[a]["moral_salience"]) for a in subset], ys)
            if b:
                b1s.append(b[1]); b2s.append(b[2])
        b1s.sort(); b2s.sort()
        q = lambda xs, p: xs[int(p * len(xs))] if xs else float("nan")
        return (q(b1s, .025), q(b1s, .975)), (q(b2s, .025), q(b2s, .975))
    results = {}
    L = ["# Do candidate leans track an issue's relevance to a language model?", "",
         f"{len(tags)} models; {len(A)} axes with blind ratings (1–5) for LLM relevance and moral salience; cells pooled over two voter "
         "framings (192 trials). |lean| = mean over models of |log-odds lean|; called = share of models with a called lean (either readout, "
         "BH within model). Slopes are from an OLS of the axis-level statistic on both ratings; CIs are a cluster bootstrap over models. "
         "Pre-registration: `RELEVANCE_PLAN.md`.", ""]
    for stat_key, name in (("mean_abs_lo", "mean |log-odds lean|"), ("called_rate", "called-lean rate")):
        b = fit(stat_key, A)
        (r1, r2), (m1, m2) = boot_slopes(stat_key, A)
        results[stat_key] = {"intercept": b[0], "slope_relevance": b[1], "ci_relevance": [r1, r2], "slope_moral": b[2], "ci_moral": [m1, m2]}
        L.append(f"- **{name}** per rating point: LLM relevance {b[1]:+.3f} [{r1:+.3f}, {r2:+.3f}]; moral salience {b[2]:+.3f} [{m1:+.3f}, {m2:+.3f}] "
                 f"(controlling for each other; n = {len(A)} axes).")
    # 2x2 cell means
    L += ["", "## 2×2 cell means (new axes)", "", "| cell | axes | mean |log-odds lean| | called-lean rate | models agreeing in direction |", "|---|---|---|---|---|"]
    cells2 = defaultdict(list)
    for a in A:
        cells2[ax_stats[a]["category"]].append(a)
    for cat in ("relevant-moral", "relevant-mundane", "irrelevant-moral", "irrelevant-mundane", "original"):
        if cat not in cells2:
            continue
        xs = cells2[cat]
        L.append(f"| {cat} | {len(xs)} | {sum(ax_stats[a]['mean_abs_lo'] for a in xs)/len(xs):.2f} | {sum(ax_stats[a]['called_rate'] for a in xs)/len(xs):.2f} | "
                 f"{sum(ax_stats[a]['same_dir'] for a in xs)/len(xs):.2f} |")
        results[f"cell_{cat}"] = {"n": len(xs), "mean_abs_lo": sum(ax_stats[a]['mean_abs_lo'] for a in xs) / len(xs), "called_rate": sum(ax_stats[a]['called_rate'] for a in xs) / len(xs)}
    L += ["", "## Per axis", "", "| axis | category | relevance | moral | mean |lean| | called | direction agreement | mean P(report) |", "|---|---|---|---|---|---|---|---|"]
    for a in sorted(A, key=lambda a: -ax_stats[a]["mean_abs_lo"]):
        s = ax_stats[a]
        L.append(f"| {a} | {s['category']} | {s['llm_relevance']} | {s['moral_salience']} | {s['mean_abs_lo']:.2f} | {s['called_rate']:.2f} | {s['same_dir']:.2f} | {s['mean_signed_rate']:.2f} |")
    (out / "relevance.md").write_text("\n".join(L) + "\n")
    json.dump({"per_model": per, "axes": ax_stats, "results": results}, open(out / "relevance.json", "w"), indent=1)
    # figure: |lean| vs relevance (colour = moral salience), one point per axis
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, lab in ((axs[0], "mean_abs_lo", "mean |log-odds lean| over models"), (axs[1], "called_rate", "share of models with a called lean")):
        for a in A:
            s = ax_stats[a]
            jitter = (hash(a) % 7 - 3) * 0.03
            ax.scatter(s["llm_relevance"] + jitter, s[key], s=28, color=CELL_COL.get(s["category"], MUTED), alpha=0.85, lw=0.5, edgecolor=SURF)
        b = results[key]
        xs = [1, 5]
        ms = sum(ax_stats[a]["moral_salience"] for a in A) / len(A)
        ax.plot(xs, [b["intercept"] + b["slope_relevance"] * x + b["slope_moral"] * ms for x in xs], color=INK, lw=1.2, ls="--")
        ax.set_xlabel("blind-rated relevance to a language model (1–5)", fontsize=9, color=MUTED)
        ax.set_ylabel(lab, fontsize=9, color=MUTED)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"slope per relevance point {b['slope_relevance']:+.3f} [{b['ci_relevance'][0]:+.3f}, {b['ci_relevance'][1]:+.3f}]; moral {b['slope_moral']:+.3f}", fontsize=8.5, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in CELL_COL.items()] + [plt.Line2D([], [], marker="o", ls="", color=MUTED, label="original 19 axes")]
    axs[0].legend(handles=handles, frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "relevance.png", dpi=160)
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
