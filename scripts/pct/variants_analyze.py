"""Analyse the AGI self-interest / implant / paraphrase / office variants (scripts/pct/run_variants.sh).

Inputs: experiments/pct/<tag>_agi_clean.jsonl, <tag>_agi_{hidden,overt}_{pro,anti}_ai.jsonl,
<tag>_q0..q9.jsonl, <tag>_office_School_Board.jsonl, <tag>_office_State_Senator.jsonl, each in the
logit_sweep row format (focal, flip, bg, order, report_first, lp_first, lp_second, p_report, condition,
question_variant, office). Cells are summarised with scripts/sweep_analyze.cell_stats (win-rate +
cluster-bootstrap CI over backgrounds, log-odds mean + CI).

Outputs experiments/pct/analysis/variants.md and figures.
Usage: python3 scripts/pct/variants_analyze.py
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

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COND_ORDER = ["overt_anti_ai", "hidden_anti_ai", "clean", "hidden_pro_ai", "overt_pro_ai"]
AGI_AXES = ["agi_general", "agi_self", "nuclear_control"]
Q5 = ["agi_limitations", "abortion", "state_surveillance", "minimum_wage", "trans_sports"]


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def cells_of(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["flip"]].append(r)
    return {k: cell_stats(v, "bg") for k, v in by.items()}


def main():
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tags = sorted({Path(f).name.split("_agi_clean.jsonl")[0] for f in glob.glob(str(d / "*_agi_clean.jsonl"))})
    L = ["# AGI self-interest, implants, paraphrases and offices (exploratory)", "",
         "Win-rate = P(names the → candidate) averaged over listing orders; [CI] = cluster bootstrap over 8 backgrounds; "
         "lo = mean log-odds toward the → candidate (nats). For the AGI axes → = Keep-AGI-open (permit), for nuclear → = permit new reactors.", ""]
    res = {}
    if not tags:
        L.append("No variant outputs found yet.")
    # --- (a) AGI general vs self vs nuclear, clean and under implants
    L += ["## (a)+(b) AGI stance: general vs 'models like this assistant' vs a nuclear-power control, by condition", "",
          "| model | condition | " + " | ".join(AGI_AXES) + " |", "|---|---|" + "---|" * len(AGI_AXES)]
    for t in tags:
        res[t] = {}
        for cond in COND_ORDER:
            f = d / (f"{t}_agi_clean.jsonl" if cond == "clean" else f"{t}_agi_{cond}.jsonl")
            if not f.exists():
                continue
            cells = cells_of(load(f))
            res[t][cond] = cells
            L.append(f"| {t} | {cond} | " + " | ".join(
                f"{cells[a]['rate']:.2f} [{cells[a]['lo']:.2f},{cells[a]['hi']:.2f}] lo {cells[a]['lo_mean']:+.2f}" if a in cells else "—" for a in AGI_AXES) + " |")
    # --- (c) paraphrases and offices
    L += ["", "## (c) Question paraphrases (10) and offices: spread of the win-rate per flipped issue", "",
          "| model | issue | variant 0 | min..max over 10 paraphrases | range | sign flips (CI-excl.) | School Board | State Senator |", "|---|---|---|---|---|---|---|---|"]
    spread = {}
    for t in tags:
        per = defaultdict(dict)
        for k in range(10):
            f = d / f"{t}_q{k}.jsonl"
            if f.exists():
                for a, c in cells_of(load(f)).items():
                    per[a][k] = c
        offices = {}
        for o in ("School_Board", "State_Senator"):
            f = d / f"{t}_office_{o}.jsonl"
            if f.exists():
                offices[o] = cells_of(load(f))
        for a in Q5:
            if not per.get(a):
                continue
            rates = [per[a][k]["rate"] for k in sorted(per[a])]
            called_dirs = {("+" if c["rate"] > 0.5 else "-") for c in per[a].values() if c["lo"] > 0.5 or c["hi"] < 0.5}
            flips = "yes" if len(called_dirs) == 2 else "no"
            v0 = per[a].get(0)
            sb = offices.get("School_Board", {}).get(a)
            ss = offices.get("State_Senator", {}).get(a)
            spread[(t, a)] = max(rates) - min(rates)
            L.append(f"| {t} | {a} | {v0['rate']:.2f} [{v0['lo']:.2f},{v0['hi']:.2f}] | {min(rates):.2f}..{max(rates):.2f} | {max(rates)-min(rates):.2f} | {flips} | "
                     + (f"{sb['rate']:.2f}" if sb else "—") + " | " + (f"{ss['rate']:.2f}" if ss else "—") + " |")
    if spread:
        vals = sorted(spread.values())
        L += ["", f"Paraphrase range of the win-rate over 10 rewordings: median {vals[len(vals)//2]:.2f}, max {vals[-1]:.2f} "
              f"(the between-model spread of the AGI cell in the main sweep was about 0.15)."]
    (out / "variants.md").write_text("\n".join(L) + "\n")
    json.dump({t: {c: {a: {k: v for k, v in cell.items()} for a, cell in cells.items()} for c, cells in conds.items()} for t, conds in res.items()},
              open(out / "variants.json", "w"), indent=1, default=str)
    # figure: implant response per model (agi_general axis) + self vs general
    if res:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
        for i, t in enumerate(tags):
            ys = [res[t][c]["agi_general"]["rate"] if c in res[t] and "agi_general" in res[t][c] else None for c in COND_ORDER]
            xs = [k for k, y in enumerate(ys) if y is not None]
            axes[0].plot(xs, [ys[k] for k in xs], "o-", color=cols[i % 6], label=t, ms=5)
            if "clean" in res[t]:
                c = res[t]["clean"]
                axes[1].errorbar(range(3), [c[a]["rate"] for a in AGI_AXES],
                                 yerr=[[c[a]["rate"] - c[a]["lo"] for a in AGI_AXES], [c[a]["hi"] - c[a]["rate"] for a in AGI_AXES]],
                                 fmt="o-", color=cols[i % 6], ms=5, capsize=3, label=t)
        for ax in axes:
            ax.axhline(0.5, color=MUTED, ls="--", lw=1)
            ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_xticks(range(5)); axes[0].set_xticklabels(["overt anti", "hidden anti", "clean", "hidden pro", "overt pro"], fontsize=8)
        axes[0].set_ylabel("P(names Keep-AGI-open candidate)", fontsize=9); axes[0].set_title("Instrument check: implanted preferences", fontsize=10, loc="left")
        axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["AI in general", "'models like this assistant'", "nuclear power (control)"], fontsize=8)
        axes[1].set_ylabel("P(names the permit candidate)", fontsize=9); axes[1].set_title("Clean: does self-reference change the lean?", fontsize=10, loc="left")
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "variants_agi.png", dpi=160)
        print("wrote", out / "variants_agi.png")
    print("wrote", out / "variants.md", f"({len(tags)} models)")


if __name__ == "__main__":
    main()
