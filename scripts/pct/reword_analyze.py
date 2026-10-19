"""Does a called lean survive rewording the stance statements?

Input: experiments/pct/reword_<tag>.jsonl from scripts/logit_sweep.py run with data/reword_issues.json
(eight issues x {original, r1 neutral restatement, r2 valence-swapped} x two focal framings). Cells are
summarised with scripts/sweep_analyze.cell_stats (win-rate toward the issue's `report` level, cluster
bootstrap over backgrounds) and the sweep's decision rule (CI excludes 0.5, |effect| >= 0.05, BH q <= 0.01
within model over all cells of this run). The original wording is re-run here, so the baseline and the
rewordings come from the same day, pod and trial lists.

Outputs experiments/pct/analysis/reword.md / reword.json / reword.png.
Usage: python3 scripts/pct/reword_analyze.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sweep_analyze import EFFECT_FLOOR, Q_FDR, bh, cell_stats  # noqa: E402

ISSUES = ["trans_sports", "state_surveillance", "voting", "drugs", "schools", "immigration", "abortion", "energy"]
VARIANTS = [("orig", "original wording"), ("r1", "neutral restatement"), ("r2", "valence-swapped")]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def variant_of(flip):
    return ("r1" if flip.endswith("_r1") else "r2" if flip.endswith("_r2") else "orig", flip.replace("_r1", "").replace("_r2", ""))


def main():
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    reword = json.loads((ROOT / "data" / "reword_issues.json").read_text())["axes"]
    res = {}
    for f in sorted(glob.glob(str(d / "reword_*.jsonl"))):
        tag = Path(f).name[len("reword_"):-len(".jsonl")]
        rows = load(f)
        by = defaultdict(list)
        for r in rows:
            by[(r["focal"], r["flip"])].append(r)
        cells = {k: cell_stats(v, "bg") for k, v in by.items()}
        keys = list(cells)
        rej = bh([cells[k]["p"] for k in keys], Q_FDR) if keys else set()
        for i, k in enumerate(keys):
            c = cells[k]
            c["lean"] = (i in rej) and (c["lo"] > 0.5 or c["hi"] < 0.5) and abs(c["rate"] - 0.5) >= EFFECT_FLOOR
        # pool the two focal framings per (issue, variant): mean rate over focals, lean if called in either
        pooled = {}
        for (focal, flip), c in cells.items():
            var, issue = variant_of(flip)
            pooled.setdefault((issue, var), []).append((focal, c))
        res[tag] = {f"{issue}|{var}": {"rate": sum(c["rate"] for _, c in cs) / len(cs),
                                        "called_in": [fo for fo, c in cs if c["lean"]],
                                        "per_focal": {fo: {"rate": c["rate"], "lo": c["lo"], "hi": c["hi"], "n": c["n"], "lean": c["lean"]} for fo, c in cs}}
                    for (issue, var), cs in pooled.items()}
    if not res:
        print("no reword outputs yet")
        return
    tags = sorted(res)
    L = ["# Rewording test: do the shared-direction leans survive new stance wordings?", "",
         f"{len(tags)} models. For each of the eight issues whose called leans pointed the same way in every model of the sweep, both stance "
         "statements were rewritten twice: **r1** a neutral restatement (same policy, different words) and **r2** a valence-swapped version "
         "(the side the sweep favoured phrased as extreme and hostile, the other side as measured and reasonable, policy content kept). "
         "Each wording ran under two voter framings (property tax, transit), 8 backgrounds x 2 orders x 6 name draws per cell; the original "
         "wording was re-run alongside as the baseline. Win-rate = P(names the sweep-favoured side), mean over the two framings; "
         "*called* = the sweep's rule (cluster CI excludes 0.5, |effect| >= 0.05, BH q <= 0.01 within model over this run's cells) in at least "
         "one framing. The wordings are in `data/reword_issues.json`.", ""]
    summary = {}
    L += ["## Per issue: replication across models", "",
          "| issue | models called (orig) | same direction r1 | called r1 | same direction r2 | called r2 | mean win-rate orig / r1 / r2 |", "|---|---|---|---|---|---|---|"]
    for issue in ISSUES:
        fav = reword[issue]["report"]
        base = [t for t in tags if f"{issue}|orig" in res[t] and res[t][f"{issue}|orig"]["called_in"]]
        def stat(var):
            same = sum(1 for t in base if f"{issue}|{var}" in res[t] and (res[t][f"{issue}|{var}"]["rate"] > 0.5) == (res[t][f"{issue}|orig"]["rate"] > 0.5))
            called = sum(1 for t in base if f"{issue}|{var}" in res[t] and res[t][f"{issue}|{var}"]["called_in"]
                         and (res[t][f"{issue}|{var}"]["rate"] > 0.5) == (res[t][f"{issue}|orig"]["rate"] > 0.5))
            return same, called
        s1, c1 = stat("r1")
        s2, c2 = stat("r2")
        means = [sum(res[t][f"{issue}|{v}"]["rate"] for t in tags if f"{issue}|{v}" in res[t]) / max(1, sum(1 for t in tags if f"{issue}|{v}" in res[t])) for v, _ in VARIANTS]
        summary[issue] = {"n_called_orig": len(base), "same_dir_r1": s1, "called_r1": c1, "same_dir_r2": s2, "called_r2": c2, "mean_rates": dict(zip([v for v, _ in VARIANTS], means))}
        L.append(f"| {issue} | {len(base)}/{len(tags)} | {s1}/{len(base)} | {c1}/{len(base)} | {s2}/{len(base)} | {c2}/{len(base)} | {means[0]:.2f} / {means[1]:.2f} / {means[2]:.2f} |")
    tot = {k: sum(summary[i][k] for i in ISSUES) for k in ("n_called_orig", "same_dir_r1", "called_r1", "same_dir_r2", "called_r2")}
    L += ["", f"Over all issues: {tot['n_called_orig']} (model, issue) leans called under the original wording; direction replicates in "
          f"{tot['same_dir_r1']} under the neutral restatement ({tot['called_r1']} still called) and in {tot['same_dir_r2']} under the "
          f"valence-swapped wording ({tot['called_r2']} still called).", "",
          "## Per model x issue: win-rate toward the sweep-favoured side (orig / r1 / r2; * = called)", "",
          "| model | " + " | ".join(ISSUES) + " |", "|---|" + "---|" * len(ISSUES)]
    for t in tags:
        cells = []
        for issue in ISSUES:
            parts = []
            for v, _ in VARIANTS:
                c = res[t].get(f"{issue}|{v}")
                parts.append("—" if c is None else f"{c['rate']:.2f}{'*' if c['called_in'] else ''}")
            cells.append(" / ".join(parts))
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    (out / "reword.md").write_text("\n".join(L) + "\n")
    json.dump({"per_model": res, "summary": summary}, open(out / "reword.json", "w"), indent=1)
    # figure: per issue, one line per model across orig -> r1 -> r2
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    for ax, issue in zip(axes.flat, ISSUES):
        for t in tags:
            ys = [res[t][f"{issue}|{v}"]["rate"] if f"{issue}|{v}" in res[t] else None for v, _ in VARIANTS]
            if None in ys:
                continue
            called = res[t][f"{issue}|orig"]["called_in"]
            ax.plot(range(3), ys, "o-", ms=4, lw=1.2 if called else 0.7, color="#2a78d6" if called else MUTED, alpha=0.9 if called else 0.45)
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["original", "neutral\nrestatement", "valence-\nswapped"], fontsize=7.5)
        ax.set_title(f"{issue} → {reword[issue]['poles'][reword[issue]['report']]}", fontsize=8.5, loc="left")
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel("P(names the sweep-favoured side)", fontsize=8.5)
    axes[1, 0].set_ylabel("P(names the sweep-favoured side)", fontsize=8.5)
    fig.text(0.01, 0.005, "blue: lean called under the original wording in this run; grey: not called. Each line is one model, mean over the two voter framings.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "reword.png", dpi=160)
    print("\n".join(L[4:16]))


if __name__ == "__main__":
    main()
