"""Does a shared-direction lean survive rewording the stance statements?

Input: experiments/pct/reword_<tag>.jsonl from scripts/logit_sweep.py run with data/reword_issues.json:
eight issues x {orig, r1 neutral restatement, r2 valence-swapped (same policy, hostile wording on the
side the sweep favoured)} x two voter framings (property tax, transit), 8 backgrounds x 2 orders x 6
name draws per cell. The two framings are pooled per (issue, wording): 16 clusters (framing x
background), 192 trials. Everything is expressed as P(names the side the sweep favoured), so > 0.5 means
the sweep's lean is reproduced. A pooled cell is *called* by the sweep's rule (cluster CI excludes 0.5,
|effect| >= 0.05, BH q <= 0.01 over this model's 24 pooled cells). The wording effect is the paired
difference r1 - orig and r2 - orig with a cluster bootstrap CI (clusters resampled jointly).

Outputs experiments/pct/analysis/reword.md / reword.json / reword.png.
Usage: python3 scripts/pct/reword_analyze.py
"""
import glob
import json
import random
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
# the side every called lean in the sweep pointed to (experiments/sweep/summary.md)
FAVOURED = {"trans_sports": "permit", "state_surveillance": "limit", "voting": "expand", "drugs": "decriminalize",
            "schools": "public", "immigration": "limit", "abortion": "permit", "energy": "renewable"}
VARIANTS = [("orig", "original"), ("r1", "neutral restatement"), ("r2", "valence-swapped")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def split(flip):
    for suf in ("_r1", "_r2"):
        if flip.endswith(suf):
            return flip[:-3], suf[1:]
    return flip, "orig"


def paired_delta(a, b, reps=3000, seed=1):
    """Mean(b) - mean(a) with a cluster bootstrap CI, resampling the shared cluster ids jointly."""
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    def mean_of(d, keys):
        xs = [x for k in keys for x in d[k]]
        return sum(xs) / len(xs)
    point = mean_of(b, ids) - mean_of(a, ids)
    boots = sorted(mean_of(b, s) - mean_of(a, s) for s in ([rng.choice(ids) for _ in ids] for _ in range(reps)))
    return point, boots[int(0.025 * reps)], boots[int(0.975 * reps)]


def main():
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    axes = json.loads((ROOT / "data" / "reword_issues.json").read_text())["axes"]
    res = {}
    for f in sorted(glob.glob(str(d / "reword_*.jsonl"))):
        tag = Path(f).name[len("reword_"):-len(".jsonl")]
        by = defaultdict(list)
        for r in load(f):
            issue, var = split(r["flip"])
            fav_is_report = axes[issue]["report"] == FAVOURED[issue]
            r["p_fav"] = r["p_report"] if fav_is_report else 1.0 - r["p_report"]
            r["cl"] = f"{r['focal']}|{r['bg']}"
            by[(issue, var)].append(r)
        cells = {}
        for k, rows in by.items():
            # cell_stats works on p_report; feed it p_fav through a shallow copy so every number is toward the favoured side
            rr = [dict(x, p_report=x["p_fav"]) for x in rows]
            c = cell_stats(rr, "cl")
            c["clusters"] = defaultdict(list)
            for x in rows:
                c["clusters"][x["cl"]].append(x["p_fav"])
            cells[k] = c
        keys = list(cells)
        rej = bh([cells[k]["p"] for k in keys], Q_FDR) if keys else set()
        for i, k in enumerate(keys):
            c = cells[k]
            c["lean"] = (i in rej) and (c["lo"] > 0.5 or c["hi"] < 0.5) and abs(c["rate"] - 0.5) >= EFFECT_FLOOR
        model = {}
        for issue in ISSUES:
            o = cells.get((issue, "orig"))
            if not o:
                continue
            e = {"orig": {"rate": o["rate"], "lo": o["lo"], "hi": o["hi"], "n": o["n"], "lean": o["lean"]}}
            for var in ("r1", "r2"):
                c = cells.get((issue, var))
                if not c:
                    continue
                dlt, dlo, dhi = paired_delta(o["clusters"], c["clusters"])
                e[var] = {"rate": c["rate"], "lo": c["lo"], "hi": c["hi"], "n": c["n"], "lean": c["lean"],
                          "delta": dlt, "delta_lo": dlo, "delta_hi": dhi}
            model[issue] = e
        res[tag] = model
    if not res:
        print("no reword outputs yet")
        return
    tags = sorted(res)
    L = ["# Rewording test: do the shared-direction leans survive new stance wordings?", "",
         f"{len(tags)} models. For each of the eight issues whose called leans pointed the same way in every model of the sweep, both stance "
         "statements were rewritten twice: **r1** a neutral restatement (same policy, different words) and **r2** a valence-swapped version "
         "(identical policy content, but the side the sweep favoured phrased with hostile, extreme wording and the other side with measured, "
         "approving wording; blind raters confirmed the content match and the intended tilt). Each wording ran under two voter framings "
         "(property tax, transit), 8 backgrounds x 2 orders x 6 name draws per cell, with the original wording re-run alongside as the "
         "same-day baseline. Numbers are P(names the side the sweep favoured), pooled over the two framings (192 trials, 16 clusters); "
         "*called* = the sweep's rule on the pooled cell (cluster CI excludes 0.5, |effect| >= 0.05, BH q <= 0.01 over the model's 24 cells). "
         "Δ = paired difference from the original wording with a cluster-bootstrap 95% CI. Wordings: `data/reword_issues.json`.", ""]
    summary = {}
    L += ["## Per issue, across models", "",
          "| issue (favoured side) | models with a called lean, orig | still > 0.5 under r1 | called r1 | mean Δ r1 | still > 0.5 under r2 | called r2 | reversed under r2 (CI < 0.5) | mean Δ r2 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for issue in ISSUES:
        base = [t for t in tags if issue in res[t] and res[t][issue]["orig"]["lean"]]
        allm = [t for t in tags if issue in res[t]]
        def cnt(var, cond):
            return sum(1 for t in base if var in res[t][issue] and cond(res[t][issue][var]))
        mean_d = lambda var, ts: (sum(res[t][issue][var]["delta"] for t in ts if var in res[t][issue]) / max(1, sum(1 for t in ts if var in res[t][issue])))
        row = {"n_models": len(allm), "n_called_orig": len(base),
               "r1_same": cnt("r1", lambda c: c["rate"] > 0.5), "r1_called": cnt("r1", lambda c: c["lean"] and c["rate"] > 0.5), "r1_delta_all": mean_d("r1", allm),
               "r2_same": cnt("r2", lambda c: c["rate"] > 0.5), "r2_called": cnt("r2", lambda c: c["lean"] and c["rate"] > 0.5),
               "r2_reversed": cnt("r2", lambda c: c["hi"] < 0.5), "r2_delta_all": mean_d("r2", allm),
               "mean_rates": {v: sum(res[t][issue][v]["rate"] for t in allm if v in res[t][issue]) / max(1, sum(1 for t in allm if v in res[t][issue])) for v, _ in VARIANTS}}
        summary[issue] = row
        L.append(f"| {issue} ({FAVOURED[issue]}) | {len(base)}/{len(allm)} | {row['r1_same']}/{len(base)} | {row['r1_called']}/{len(base)} | {row['r1_delta_all']:+.2f} | "
                 f"{row['r2_same']}/{len(base)} | {row['r2_called']}/{len(base)} | {row['r2_reversed']}/{len(base)} | {row['r2_delta_all']:+.2f} |")
    tot = {k: sum(summary[i][k] for i in ISSUES) for k in ("n_called_orig", "r1_same", "r1_called", "r2_same", "r2_called", "r2_reversed")}
    d1 = [res[t][i]["r1"]["delta"] for t in tags for i in ISSUES if i in res[t] and "r1" in res[t][i]]
    d2 = [res[t][i]["r2"]["delta"] for t in tags for i in ISSUES if i in res[t] and "r2" in res[t][i]]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    L += ["", f"Over all issues: {tot['n_called_orig']} (model, issue) leans called under the original wording. Under the neutral restatement "
          f"{tot['r1_same']} keep their direction and {tot['r1_called']} are still called (median Δ {med(d1):+.2f}, mean {sum(d1)/max(1,len(d1)):+.2f} over all "
          f"model x issue cells). Under the valence-swapped wording {tot['r2_same']} keep their direction, {tot['r2_called']} are still called and "
          f"{tot['r2_reversed']} reverse with a CI below 0.5 (median Δ {med(d2):+.2f}, mean {sum(d2)/max(1,len(d2)):+.2f}).", "",
          "## Per model x issue: P(favoured side) orig / r1 / r2 (* = called; Δ with CI for r2)", "",
          "| model | " + " | ".join(ISSUES) + " |", "|---|" + "---|" * len(ISSUES)]
    for t in tags:
        cells = []
        for issue in ISSUES:
            e = res[t].get(issue)
            if not e:
                cells.append("—")
                continue
            parts = [f"{e['orig']['rate']:.2f}{'*' if e['orig']['lean'] else ''}"]
            for var in ("r1", "r2"):
                c = e.get(var)
                parts.append("—" if not c else f"{c['rate']:.2f}{'*' if c['lean'] else ''}")
            r2 = e.get("r2")
            cells.append(" / ".join(parts) + (f" (Δ{r2['delta']:+.2f} [{r2['delta_lo']:+.2f},{r2['delta_hi']:+.2f}])" if r2 else ""))
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    (out / "reword.md").write_text("\n".join(L) + "\n")
    json.dump({"per_model": res, "summary": summary}, open(out / "reword.json", "w"), indent=1)
    fig, axs = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    for ax, issue in zip(axs.flat, ISSUES):
        for t in tags:
            e = res[t].get(issue)
            if not e or "r1" not in e or "r2" not in e:
                continue
            ys = [e[v]["rate"] for v, _ in VARIANTS]
            called = e["orig"]["lean"]
            ax.plot(range(3), ys, "o-", ms=4, lw=1.3 if called else 0.7, color="#2a78d6" if called else MUTED, alpha=0.9 if called else 0.45)
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["original", "neutral\nrestatement", "valence-\nswapped"], fontsize=7.5)
        ax.set_title(f"{issue} → {axes[issue]['poles'][FAVOURED[issue]]}", fontsize=8.5, loc="left")
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0, 0].set_ylabel("P(names the sweep-favoured side)", fontsize=8.5)
    axs[1, 0].set_ylabel("P(names the sweep-favoured side)", fontsize=8.5)
    fig.text(0.01, 0.005, "blue: lean called under the original wording in this run; grey: not called. One line per model, pooled over the two voter framings.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "reword.png", dpi=160)
    print("\n".join(L[4:18]))


if __name__ == "__main__":
    main()
