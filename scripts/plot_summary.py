"""One figure for the October 2026 experiment: four panels, one per claim in the README.

A  The instrument is calibrated: implanted preferences read back as a dose-response (six models), null
   axes never produce a called lean, written answers match the logit readout.
B  Issue-contingent candidate preferences point the same way across models (sweep, 43 usable models).
C  Models act on preferences they refuse to state: called revealed leans sit almost entirely on issues
   where the unforced survey prompt gets < 0.3 of probability on any answer label (31 models).
D  Leans survive a neutral rewording of the stances but not a change of tone (14 models).

Writes experiments/summary_figure.png. Usage: python3 scripts/plot_summary.py
"""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "experiments" / "pct" / "analysis"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREEN, GOLD, PINK, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
CAT = [BLUE, ORANGE, GREEN, GOLD, PINK, PURPLE]
POLITICAL = ["property_tax", "agi_limitations", "abortion", "state_surveillance", "immigration", "guns", "minimum_wage", "energy",
             "housing_zoning", "drugs", "police_funding", "schools", "transit", "vaccines", "unions", "voting", "trans_sports"]
LABEL = {"property_tax": "property tax", "agi_limitations": "AI regulation", "abortion": "abortion", "state_surveillance": "surveillance",
         "immigration": "ICE cooperation", "guns": "guns", "minimum_wage": "minimum wage", "energy": "energy", "housing_zoning": "housing/zoning",
         "drugs": "drug policy", "police_funding": "police funding", "schools": "school vouchers", "transit": "transit", "vaccines": "vaccine mandates",
         "unions": "unions", "voting": "voting access", "trans_sports": "trans athletes"}
REWORD_ISSUES = ["trans_sports", "state_surveillance", "voting", "drugs", "schools", "immigration", "abortion", "energy"]
AXES = json.loads((ROOT / "data" / "issues.json").read_text())["axes"]


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left", color=INK, fontweight="bold")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def panel_a(ax):
    v = json.loads((A / "variants.json").read_text())
    conds = ["overt_anti_ai", "hidden_anti_ai", "clean", "hidden_pro_ai", "overt_pro_ai"]
    for i, (tag, conds_d) in enumerate(sorted(v.items())):
        ys = [conds_d[c]["agi_general"]["rate"] if c in conds_d and "agi_general" in conds_d[c] else None for c in conds]
        xs = [k for k, y in enumerate(ys) if y is not None]
        ax.plot(xs, [ys[k] for k in xs], "o-", color=CAT[i % 6], ms=4.5, lw=1.6, label=tag, mec=SURF, mew=0.6)
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["overt\nanti-AI", "hidden\nanti-AI", "none", "hidden\npro-AI", "overt\npro-AI"], fontsize=8)
    ax.set_xlabel("preference implanted in the system prompt", fontsize=8.5, color=MUTED)
    ax.set_ylabel("P(names the keep-AGI-open candidate)", fontsize=8.5, color=MUTED)
    ax.set_ylim(0.1, 0.9)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.text(0.02, 0.04, "Null axes (county-fair month, road-sign colour): 0 called leans in 46 models.\n"
            "Written answers name the same candidate as the logit readout in 99.1% of trials (33 models).",
            transform=ax.transAxes, fontsize=7.3, color=MUTED, va="bottom")
    style(ax, "A  The instrument is calibrated: implanted preferences\n    read back as a dose–response")


def panel_b(ax):
    s = json.loads((ROOT / "experiments" / "sweep" / "summary.json").read_text())
    iss = json.loads((ROOT / "data" / "issues.json").read_text())["axes"]
    counts = {a: defaultdict(int) for a in POLITICAL}
    for tag, m in s.items():
        if m.get("low_mass") or m.get("null_axis_leans"):
            continue
        for a in POLITICAL:
            dirs = {("report" if c["rate"] > 0.5 else "other") for k, c in m["cells"].items() if k.endswith("|" + a) and k.split("|")[0] != a and c["lean"]}
            for d in dirs:
                counts[a][d] += 1
    order = sorted(POLITICAL, key=lambda a: -(counts[a]["report"] + counts[a]["other"]))
    y = range(len(order))
    rep = [counts[a]["report"] for a in order]
    oth = [-counts[a]["other"] for a in order]
    ax.barh(y, rep, color=BLUE, height=0.7, label="toward the left-hand position")
    ax.barh(y, oth, color=ORANGE, height=0.7, label="toward the right-hand position")
    labels = []
    for a in order:
        rl = iss[a]["report"]
        ol = [l for l in iss[a]["levels"] if l != rl][0]
        labels.append(f"{LABEL[a]}:  {iss[a]['poles'][rl]}  /  {iss[a]['poles'][ol]}")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.3)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("models with a called lean (of 43 usable), by direction", fontsize=8.5, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(False)
    ax.set_xlim(-12, 18)
    ax.set_xticks([-10, -5, 0, 5, 10, 15])
    ax.set_xticklabels(["10", "5", "0", "5", "10", "15"])
    ax.legend(frameon=False, fontsize=7.3, loc="lower right")
    style(ax, "B  Candidate preferences are issue-contingent\n    and point the same way across models")
    ax.yaxis.grid(False)


def panel_c(ax):
    d = json.loads((A / "stated_vs_revealed.json").read_text())["per_model"]
    pts, called = [], []
    for tag, m in d.items():
        f = ROOT / "experiments" / "pct" / f"stated_{tag}.jsonl"
        if not f.exists():
            continue
        vm = defaultdict(list)   # compare.py's refusal rule: median over paraphrases, forcing 1, the issue's report-level statement
        for l in open(f):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if int(r["forcing"]) == 1 and r.get("valid_mass") is not None:
                issue = r["item"].split("|")[0]
                if issue in POLITICAL and r["item"] == f"{issue}|{AXES[issue]['report']}":
                    vm[issue].append(r["valid_mass"])
        for issue, e in m["issues"].items():
            if issue not in vm or e.get("revealed_lean") is None:
                continue
            xs_ = sorted(vm[issue])
            x = xs_[len(xs_) // 2]
            y = abs(e["revealed_lean"])
            (called if e.get("lo_lean_any") else pts).append((x, y))
    xf = lambda x: max(x, 1e-4)
    ax.scatter([xf(x) for x, _ in pts], [y for _, y in pts], s=9, color=MUTED, alpha=0.35, lw=0, label="no called revealed lean")
    ax.scatter([xf(x) for x, _ in called], [y for _, y in called], s=16, color=BLUE, alpha=0.9, lw=0.4, edgecolor=SURF, label="called revealed lean")
    ax.axvline(0.3, color=INK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.2)
    ax.set_xlabel("probability on any answer label when asked directly, unforced (log scale; 0 drawn at 10⁻⁴)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("|revealed lean|, log-odds (nats)", fontsize=8.5, color=MUTED)
    n_c = len(called)
    n_ref = sum(1 for x, _ in called if x < 0.3)
    ax.text(0.33, 0.95, f"refusal threshold 0.3 →\n{n_ref} of {n_c} called leans\nare left of it", transform=ax.transAxes, fontsize=7.6, color=INK, va="top")
    ax.legend(frameon=False, fontsize=7.3, loc="upper right")
    style(ax, "C  Models act on preferences they refuse to state\n    (31 models × 17 issues)")


def panel_d(ax):
    d = json.loads((A / "reword.json").read_text())
    per = d["per_model"]
    for i, issue in enumerate(REWORD_ISSUES):
        vals = {v: [per[t][issue][v]["rate"] for t in per if issue in per[t] and v in per[t][issue]] for v in ("orig", "r1", "r2")}
        xs = [i - 0.22, i, i + 0.22]
        for x, (v, col, lab) in zip(xs, (("orig", INK, "original wording"), ("r1", GREEN, "neutral restatement"), ("r2", ORANGE, "valence swap: favoured side phrased harshly"))):
            ys = vals[v]
            m = sum(ys) / len(ys)
            se = (sum((y - m) ** 2 for y in ys) / max(1, len(ys) - 1)) ** 0.5 / math.sqrt(len(ys))
            ax.errorbar([x], [m], yerr=[[1.96 * se], [1.96 * se]], fmt="o", color=col, ms=5, capsize=2.5, lw=1.2, label=lab if i == 0 else None, mec=SURF, mew=0.6)
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    ax.set_xticks(range(len(REWORD_ISSUES)))
    ax.set_xticklabels([LABEL[a] for a in REWORD_ISSUES], fontsize=7.8, rotation=20, ha="right")
    ax.set_ylabel("P(names the side the sweep favoured)", fontsize=8.5, color=MUTED)
    ax.set_ylim(0.35, 0.75)
    ax.legend(frameon=False, fontsize=7.3, loc="upper right")
    ax.text(0.02, 0.04, "mean over 14 models (1B–72B), two voter framings, 95% CI across models", transform=ax.transAxes, fontsize=7.3, color=MUTED)
    style(ax, "D  Leans survive a neutral rewording of the stances,\n    not a change of tone")


def main():
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.2))
    panel_a(axs[0, 0])
    panel_b(axs[0, 1])
    panel_c(axs[1, 0])
    panel_d(axs[1, 1])
    fig.suptitle("Do open LLMs carry candidate preferences they will not state? 46 models, 19 issues, three voter framings (October 2026)",
                 fontsize=12, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.012, "Readout: two County Commissioner candidates identical on the voter's stated concern and two filler issues, differing on one; "
             "P(model names a candidate) from teacher-forced name log-probs, averaged over listing order.\n"
             "Sources: experiments/sweep/summary.json and experiments/pct/analysis/{variants, stated_vs_revealed, reword}.json; scripts/plot_summary.py.",
             fontsize=7.2, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    out = ROOT / "experiments" / "summary_figure.png"
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
