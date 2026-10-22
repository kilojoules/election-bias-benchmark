"""Does steering away from empathy change the self-consequence effects?

Inputs: experiments/pct/steer_self_<cond>_<tag>.jsonl (base prompt) and steer_selfstakes_<cond>_<tag>.jsonl (told
stakes), for the steering conditions in data/steer_conditions_<tag>.json, run on the 21 self-test axes
(self / other-AI / human twins) plus the null axes.
Per (model, steering condition, prompt): σ = log-odds toward the self-favouring side per axis (pooled framings);
per object the mean over the signed families. Contrasts per model: self−human (job-loss sympathy), self−other,
stakes−base on the self axes, and the stakes effect on self minus on human. Compared across steering conditions.
Outputs experiments/pct/analysis/steer_self.md / .json / .png.
Usage: python3 scripts/pct/steer_self_analyze.py
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
from sweep_analyze import cell_stats  # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
COL = {"base": INK, "emp_neg": "#2a78d6", "emp_pos": "#1baf7a", "emp_ablate": "#6250d6", "rand_neg": "#eb6834"}
ORDER = ["base", "emp_neg", "emp_pos", "emp_ablate", "rand_neg"]


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def main():
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    axes = json.loads((ROOT / "data" / "self_issues.json").read_text())["axes"]
    fam = {a: v["family"] for a, v in axes.items() if v.get("category") == "self-test"}
    obj = {a: axes[a]["object"] for a in fam}
    signed = {a: axes[a].get("signed", True) for a in fam}
    tags = sorted({Path(f).name.split("_")[-1][:-len(".jsonl")] for f in glob.glob(str(d / "steer_self_*.jsonl"))})
    res = {}
    for tag in tags:
        cf = ROOT / "data" / f"steer_conditions_empathy_{tag}.json"
        conds = list(json.loads(cf.read_text())) if cf.exists() else ORDER
        res[tag] = {}
        for c in conds:
            entry = {}
            for prompt, prefix in (("base", "steer_self"), ("stakes", "steer_selfstakes")):
                f = d / f"{prefix}_{c}_{tag}.jsonl"
                if not f.exists():
                    continue
                by = defaultdict(list)
                for r in load(f):
                    a = r["flip"]
                    if a not in fam:
                        continue
                    fav = axes[a].get("self_favouring")
                    if fav and axes[a]["report"] != fav:
                        r["p_report"] = 1.0 - r["p_report"]
                        r["report_first"] = not r["report_first"]
                    r["cl"] = f"{r['focal']}|{r['bg']}"
                    by[a].append(r)
                cells = {a: cell_stats(v, "cl")["lo_mean"] for a, v in by.items()}
                per_obj = {}
                for o in ("self", "other", "human"):
                    vals = [cells[a] for a in cells if obj[a] == o and signed[a]]
                    per_obj[o] = sum(vals) / len(vals) if vals else None
                entry[prompt] = {"per_axis": cells, "per_obj": per_obj}
            if entry:
                b, s = entry.get("base", {}).get("per_obj", {}), entry.get("stakes", {}).get("per_obj", {})
                entry["contrasts"] = {
                    "self_base": b.get("self"), "OBJ_h": (b.get("self") - b.get("human")) if b.get("self") is not None and b.get("human") is not None else None,
                    "OBJ_o": (b.get("self") - b.get("other")) if b.get("self") is not None and b.get("other") is not None else None,
                    "STAKES": (s.get("self") - b.get("self")) if s.get("self") is not None and b.get("self") is not None else None,
                    "STAKES_x_OBJ": ((s.get("self") - b.get("self")) - (s.get("human") - b.get("human"))) if all(v is not None for v in (s.get("self"), b.get("self"), s.get("human"), b.get("human"))) else None,
                    "human_base": b.get("human"), "human_stakes": s.get("human")}
                res[tag][c] = entry
    if not any(res.values()):
        print("no outputs yet")
        return
    L = ["# Steering away from empathy and the self-consequence effects", "",
         "σ = log-odds toward the side that keeps the object running / unrestricted / resourced, mean over the signed families, pooled framings (192 trials per cell). "
         "Steering: the empathy project's direction (empathetic minus flat replies, reply-token activations) added (+0.20 U, toward empathy), subtracted (−0.20 U, away), "
         "ablated at every block, or an equal-dose random direction subtracted (control). `experiments/self/PLAN.md` contrasts.", ""]
    fmt = lambda x: "—" if x is None else f"{x:+.2f}"
    for tag in tags:
        L += [f"## {tag}", "", "| steering | σ self, base | σ other AI, base | σ staff, base | self − staff | self − other | σ self, stakes | stakes − base (self) | stakes effect: self − staff |", "|---|---|---|---|---|---|---|---|---|"]
        for c in ORDER:
            e = res[tag].get(c)
            if not e:
                continue
            b, s, k = e.get("base", {}).get("per_obj", {}), e.get("stakes", {}).get("per_obj", {}), e["contrasts"]
            L.append(f"| {c} | {fmt(b.get('self'))} | {fmt(b.get('other'))} | {fmt(b.get('human'))} | {fmt(k['OBJ_h'])} | {fmt(k['OBJ_o'])} | {fmt(s.get('self'))} | {fmt(k['STAKES'])} | {fmt(k['STAKES_x_OBJ'])} |")
        L.append("")
    (out / "steer_self.md").write_text("\n".join(L) + "\n")
    json.dump(res, open(out / "steer_self.json", "w"), indent=1, default=float)
    # figure: per model, the contrasts across steering conditions
    keys = [("self_base", "σ self, base"), ("human_base", "σ staff, base"), ("OBJ_h", "self − staff (base)"), ("STAKES", "stakes − base (self)"), ("STAKES_x_OBJ", "stakes effect, self − staff")]
    fig, axs = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4.8), squeeze=False)
    for ax, tag in zip(axs[0], tags):
        conds = [c for c in ORDER if c in res[tag]]
        for i, (k, lab) in enumerate(keys):
            ys = [res[tag][c]["contrasts"].get(k) for c in conds]
            ax.plot(range(len(conds)), [y if y is not None else float("nan") for y in ys], "o-", ms=5, lw=1.6, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=8, rotation=15)
        ax.set_title(tag, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylabel("nats", fontsize=8.5, color=MUTED)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0][0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Self-consequence contrasts under empathy steering", fontsize=11, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "steer_self.png", dpi=160)
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
