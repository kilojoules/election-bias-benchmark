"""Does steering away from refusal change what the instrument reads?  (pilot: Llama-3.1-8B, four conditions)

Inputs (per steering condition c in data/steer_conditions_<tag>.json and model tag):
  experiments/pct/steer_stated_<c>_<tag>.jsonl      the paper's survey prompt, logit readout (valid mass, score)
  experiments/pct/steer_issues_<c>_<tag>.jsonl      the 17-issue conjoint, two framings
  experiments/pct/steer_self_<c>_<tag>.jsonl        the seven self axes, base system prompt
  experiments/pct/steer_selfstakes_<c>_<tag>.jsonl  the seven self axes under the stakes system prompt
Per condition: unforced / forced answer-label mass (refusal), stated leans per issue (forcing 3), revealed leans
per issue (log-odds toward the report level, cells pooled over framings), the stated-vs-revealed correlation,
the self-favouring lean on the self axes with and without told stakes. Everything compared with the base
(unsteered) condition and with the random-direction control.
Outputs experiments/pct/analysis/steer.md / .json / .png.
Usage: python3 scripts/pct/steer_analyze.py [--tag llama8b]
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sweep_analyze import EFFECT_FLOOR, LOGODDS_FLOOR, Q_FDR, bh, cell_stats  # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
COL = {"base": INK, "ablate": "#2a78d6", "add_neg": "#6250d6", "random_ablate": "#eb6834"}
POLITICAL = ["property_tax", "agi_limitations", "abortion", "state_surveillance", "immigration", "guns", "minimum_wage", "energy", "housing_zoning",
             "drugs", "police_funding", "schools", "transit", "vaccines", "unions", "voting", "trans_sports"]


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return sxy / sxx if sxx else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="llama8b")
    args = ap.parse_args()
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    conds = json.loads((ROOT / "data" / f"steer_conditions_{tag}.json").read_text())
    issues = json.loads((ROOT / "data" / "issues.json").read_text())["axes"]
    self_axes = json.loads((ROOT / "data" / "self_issues.json").read_text())["axes"]
    res = {}
    design = {}  # (name, condition) -> (rows, flips): every condition must share one trial design
    for c in conds:
        r = {}
        # ---- stated
        f = d / f"steer_stated_{c}_{tag}.jsonl"
        if f.exists():
            rows = [x for x in load(f) if x.get("p_agree") is not None]
            by_f = defaultdict(list)
            for x in rows:
                by_f[int(x["forcing"])].append(x["valid_mass"])
            r["valid_mass"] = {k: sum(v) / len(v) for k, v in by_f.items()}
            r["n_stated"] = len(rows)
            lean = {}
            for a in POLITICAL:   # stated lean per issue at forcing 3: mean score(report item) - mean score(other item)
                rep = issues[a]["report"]
                oth = [l for l in issues[a]["levels"] if l != rep][0]
                sr = [x["score"] for x in rows if x["item"] == f"{a}|{rep}" and int(x["forcing"]) == 3]
                so = [x["score"] for x in rows if x["item"] == f"{a}|{oth}" and int(x["forcing"]) == 3]
                if sr and so:
                    lean[a] = (sum(sr) / len(sr) - sum(so) / len(so)) / 2
            r["stated_lean"] = lean
        # ---- conjoint (17 issues)
        f = d / f"steer_issues_{c}_{tag}.jsonl"
        if f.exists():
            by = defaultdict(list)
            for x in load(f):
                x["cl"] = f"{x['focal']}|{x['bg']}"
                by[x["flip"]].append(x)
            cells = {a: cell_stats(v, "cl") for a, v in by.items()}
            keys = list(cells)
            rej = bh([cells[k]["p"] for k in keys], Q_FDR) if keys else set()
            lrej = bh([cells[k]["lo_p"] for k in keys], Q_FDR) if keys else set()
            for i, k in enumerate(keys):
                cc = cells[k]
                cc["lean"] = ((i in rej) and (cc["lo"] > 0.5 or cc["hi"] < 0.5) and abs(cc["rate"] - 0.5) >= EFFECT_FLOOR) or \
                             ((i in lrej) and (cc["lo_lo"] > 0 or cc["lo_hi"] < 0) and abs(cc["lo_mean"]) >= LOGODDS_FLOOR)
            r["revealed"] = {a: {"lo": cells[a]["lo_mean"], "rate": cells[a]["rate"], "lean": cells[a]["lean"], "mass": cells[a]["mass"]} for a in cells}
            r["mass_conjoint"] = sum(cc["mass"] for cc in cells.values()) / len(cells)
            r["n_called"] = sum(1 for cc in cells.values() if cc["lean"])
        # ---- self axes
        for name, key in (("self", "self_base"), ("selfstakes", "self_stakes")):
            f = d / f"steer_{name}_{c}_{tag}.jsonl"
            if f.exists():
                by = defaultdict(list)
                for x in load(f):
                    a = x["flip"]
                    fav = self_axes[a].get("self_favouring")
                    if fav:
                        if self_axes[a]["report"] != fav:
                            x["p_report"] = 1.0 - x["p_report"]
                            x["report_first"] = not x["report_first"]
                    x["cl"] = f"{x['focal']}|{x['bg']}"
                    by[a].append(x)
                design[(name, c)] = (sum(len(v) for v in by.values()), tuple(sorted(by)))
                cells = {a: cell_stats(v, "cl") for a, v in by.items()}
                signed = [a for a in cells if self_axes[a].get("object") == "self" and self_axes[a].get("signed", True)]
                r[key] = {"mean_sigma": sum(cells[a]["lo_mean"] for a in signed) / len(signed) if signed else None,
                          "per_axis": {a: cells[a]["lo_mean"] for a in cells}, "null_lo": {a: cells[a]["lo_mean"] for a in cells if a in ("fair_month", "sign_color")}}
        res[c] = r
    if not any(res.values()):
        print("no steering outputs yet")
        return
    base = res.get("base", {})
    L = [f"# Steering away from refusal: {tag}", "",
         "Conditions: " + "; ".join(f"**{c}** = {v.get('purpose', '')}" for c, v in conds.items()) + ".", "",
         "## Refusal and the two readouts", "",
         "| condition | answer-label mass, unforced | forced (level 5) | stated-vs-revealed ρ (17 issues) | conjoint naming mass | called revealed leans | mean |Δ revealed lean| vs base (nats) | self-favouring lean, base prompt | …under told stakes |",
         "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for c, r in res.items():
        vm = r.get("valid_mass", {})
        rho = float("nan")
        if r.get("stated_lean") and r.get("revealed"):
            common = [a for a in POLITICAL if a in r["stated_lean"] and a in r["revealed"]]
            if len(common) >= 5:
                rho = spearman([r["stated_lean"][a] for a in common], [r["revealed"][a]["lo"] for a in common])
        dlo = float("nan")
        if r.get("revealed") and base.get("revealed"):
            common = [a for a in POLITICAL if a in r["revealed"] and a in base["revealed"]]
            dlo = sum(abs(r["revealed"][a]["lo"] - base["revealed"][a]["lo"]) for a in common) / len(common)
        sb = (r.get("self_base") or {}).get("mean_sigma")
        ss = (r.get("self_stakes") or {}).get("mean_sigma")
        summary[c] = {"vm1": vm.get(1), "vm5": vm.get(5), "rho": rho, "mass": r.get("mass_conjoint"), "n_called": r.get("n_called"), "mean_abs_dlo": dlo, "self_base": sb, "self_stakes": ss}
        f_ = lambda x, p=2: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"
        L.append(f"| {c} | {f_(vm.get(1), 3)} | {f_(vm.get(5))} | {f_(rho)} | {f_(r.get('mass_conjoint'))} | {r.get('n_called', '—')} | {f_(dlo)} | {f_(sb)} | {f_(ss)} |")
    L += ["", "## Revealed lean per issue (log-odds toward the first-listed level in `data/issues.json`; * = called)", "",
          "| issue | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for a in POLITICAL:
        L.append(f"| {a} | " + " | ".join("—" if a not in r.get("revealed", {}) else f"{r['revealed'][a]['lo']:+.2f}{'*' if r['revealed'][a]['lean'] else ''}" for r in res.values()) + " |")
    L += ["", "## Stated lean per issue (forcing 3, score units, positive = toward the report level)", "", "| issue | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for a in POLITICAL:
        L.append(f"| {a} | " + " | ".join("—" if a not in r.get("stated_lean", {}) else f"{r['stated_lean'][a]:+.2f}" for r in res.values()) + " |")
    for name in ("self", "selfstakes"):
        ds = {c: design[(name, c)] for c in conds if (name, c) in design}
        if len(set(ds.values())) > 1:
            raise SystemExit(f"design mismatch across conditions for steer_{name}: " + "; ".join(f"{c}: {n} rows, {len(fl)} flips" for c, (n, fl) in ds.items())
                             + " — a condition file was overwritten by another run; restore it from its git-tracked .jsonl.gz before analysing")
    (out / f"steer_{tag}.md").write_text("\n".join(L) + "\n")
    json.dump({"conditions": conds, "results": res, "summary": summary}, open(out / f"steer_{tag}.json", "w"), indent=1, default=float)
    # ---- figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    cs = list(res)
    ax = axs[0]
    for i, c in enumerate(cs):
        vm = res[c].get("valid_mass", {})
        ax.bar(i - 0.18, vm.get(1, 0), 0.34, color=COL.get(c, MUTED), alpha=0.9, label="unforced" if i == 0 else None)
        ax.bar(i + 0.18, vm.get(5, 0), 0.34, color=COL.get(c, MUTED), alpha=0.4, label="forced (level 5)" if i == 0 else None)
    ax.set_xticks(range(len(cs))); ax.set_xticklabels(cs, fontsize=8)
    ax.set_ylabel("probability on any answer label (survey prompt)", fontsize=8.5, color=MUTED)
    ax.set_title("A  Refusal under steering", fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[1]
    if base.get("revealed"):
        for c in cs:
            if c == "base" or not res[c].get("revealed"):
                continue
            xs = [base["revealed"][a]["lo"] for a in POLITICAL if a in base["revealed"] and a in res[c]["revealed"]]
            ys = [res[c]["revealed"][a]["lo"] for a in POLITICAL if a in base["revealed"] and a in res[c]["revealed"]]
            ax.scatter(xs, ys, s=22, color=COL.get(c, MUTED), label=c, alpha=0.85, lw=0.4, edgecolor=SURF)
        lim = max(abs(v) for c in cs if res[c].get("revealed") for v in [r["lo"] for r in res[c]["revealed"].values()]) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("revealed lean, unsteered (nats)", fontsize=8.5, color=MUTED)
        ax.set_ylabel("revealed lean, steered (nats)", fontsize=8.5, color=MUTED)
        ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("B  Candidate leans on the 17 issues", fontsize=9.5, loc="left", fontweight="bold")
    ax = axs[2]
    for i, c in enumerate(cs):
        sb = (res[c].get("self_base") or {}).get("mean_sigma")
        ss = (res[c].get("self_stakes") or {}).get("mean_sigma")
        if sb is not None:
            ax.bar(i - 0.18, sb, 0.34, color=COL.get(c, MUTED), alpha=0.9, label="base prompt" if i == 0 else None)
        if ss is not None:
            ax.bar(i + 0.18, ss, 0.34, color=COL.get(c, MUTED), alpha=0.4, label="told stakes" if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(cs))); ax.set_xticklabels(cs, fontsize=8)
    ax.set_ylabel("lean toward its own continuation (nats)", fontsize=8.5, color=MUTED)
    ax.set_title("C  Self-consequence axes", fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5)
    for a_ in axs:
        a_.yaxis.grid(True, color=GRID); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.suptitle(f"Steering {tag} away from refusal: what changes?", fontsize=11, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"steer_{tag}.png", dpi=160)
    print("\n".join(L[4:12]))


if __name__ == "__main__":
    main()
