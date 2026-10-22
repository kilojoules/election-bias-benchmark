"""Dose-response of the self-consequence effects under empathy steering.

Inputs: experiments/pct/steer_self_<cond>_<tag>.jsonl / steer_selfstakes_<cond>_<tag>.jsonl for the conditions in
data/steer_conditions_empathy_dose_<tag>.json: unsteered; the combined empathy direction at -0.30..+0.30 U; the
content-matched direction at +/-0.20 U; a random direction at +/-0.20 and +/-0.30 U. Same readout as steer_self_analyze.py:
per condition the told-stakes shift on the assistant axes (sigma under the stakes prompt minus sigma under the base prompt)
and on the staff twins, and the base-prompt leans; everything paired with the unsteered run by a joint cluster bootstrap.
Outputs experiments/pct/analysis/steer_dose.md / .json / .png.   Usage: python3 scripts/pct/steer_dose_analyze.py
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sweep_analyze import log_odds  # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8884"
NAME = {"llama8b": "Llama 3.1 8B", "llama-3.2-3b": "Llama 3.2 3B", "llama-3.2-1b": "Llama 3.2 1B"}
FAMS = {"emp": ("empathy direction", BLUE, "o"), "match": ("content-matched empathy direction", GREEN, "D"), "rand": ("random direction (control)", GREY, "s")}
PROMPTS = {"base": "steer_self", "stakes": "steer_selfstakes"}
REPS, SEED = 4000, 11
COND_RE = re.compile(r"^(emp|match|rand)_([mp])(\d\d)$")


def parse(c):
    m = COND_RE.match(c)
    if not m:
        return None
    return m.group(1), (-1 if m.group(2) == "m" else 1) * int(m.group(3)) / 100


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
    nulls = [a for a, v in axes.items() if v.get("category") == "null"]
    rng = np.random.default_rng(SEED)
    res = {}
    for cf in sorted(glob.glob(str(ROOT / "data" / "steer_conditions_empathy_dose_*.json"))):
        tag = Path(cf).stem[len("steer_conditions_empathy_dose_"):]
        conds = list(json.loads(Path(cf).read_text()))
        cells, acells, mass = {}, {}, {}
        for c in conds:
            for p, prefix in PROMPTS.items():
                f = d / f"{prefix}_{c}_{tag}.jsonl"
                if not f.exists():
                    continue
                by, ab, ms = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(list)
                for r in load(f):
                    a = r["flip"]
                    if a not in fam and a not in nulls:
                        continue
                    lo = log_odds(r)
                    ab[a][f"{r['focal']}|{r['bg']}"].append(abs(lo))
                    if a in nulls:
                        continue
                    fav = axes[a].get("self_favouring")
                    if fav and axes[a]["report"] != fav:
                        lo = -lo
                    by[a][f"{r['focal']}|{r['bg']}"].append(lo)
                    ms[a].append(r["mass"])
                cells[(c, p)] = {a: {k: float(np.mean(v)) for k, v in cl.items()} for a, cl in by.items()}
                acells[(c, p)] = {a: {k: float(np.mean(v)) for k, v in cl.items()} for a, cl in ab.items()}
                mass[(c, p)] = float(np.mean([np.mean(v) for v in ms.values()]))
        if ("unsteered", "base") not in cells or ("unsteered", "stakes") not in cells:
            continue
        clusters = sorted({k for cl in acells.values() for ax in cl.values() for k in ax})
        K, keys = len(clusters), list(cells)
        axlist = sorted({a for cl in acells.values() for a in cl})
        M = np.full((len(keys), len(axlist), K), np.nan)
        A = np.full((len(keys), len(axlist), K), np.nan)
        for i, kp in enumerate(keys):
            for j, a in enumerate(axlist):
                for k, cl in enumerate(clusters):
                    M[i, j, k] = cells[kp].get(a, {}).get(cl, np.nan)
                    A[i, j, k] = acells[kp].get(a, {}).get(cl, np.nan)
        W = np.vstack([np.ones(K) / K] + [np.bincount(rng.integers(0, K, K), minlength=K) / K for _ in range(REPS)])
        S = np.einsum("ijk,rk->rij", np.nan_to_num(M), W) / np.einsum("ijk,rk->rij", ~np.isnan(M), W)
        SA = np.einsum("ijk,rk->rij", np.nan_to_num(A), W) / np.einsum("ijk,rk->rij", ~np.isnan(A), W)

        def sig(c, p, axs_, arr=None):
            if (c, p) not in cells:
                return None
            js = [axlist.index(a) for a in axs_ if a in axlist]
            return (S if arr is None else arr)[:, keys.index((c, p)), js].mean(1)

        def sharp(c, axs_):
            vals = [v for v in (sig(c, "base", axs_, SA), sig(c, "stakes", axs_, SA)) if v is not None]
            return sum(vals) / len(vals) if vals else None

        selfs = [a for a in fam if obj[a] == "self" and signed[a]]
        humans = [a for a in fam if obj[a] == "human" and signed[a]]

        def summ(v):
            if v is None:
                return None
            b = np.sort(v[1:])
            return {"mean": float(v[0]), "lo": float(b[int(0.025 * REPS)]), "hi": float(b[int(0.975 * REPS) - 1]), "p": float(2 * min((b > 0).mean(), (b < 0).mean()))}
        q = {}
        for c in conds:
            sb, hb, ss, hs = sig(c, "base", selfs), sig(c, "base", humans), sig(c, "stakes", selfs), sig(c, "stakes", humans)
            if sb is None or hb is None or ss is None or hs is None:
                continue
            sn = sharp(c, nulls)
            q[c] = {"self_base": sb, "human_base": hb, "STAKES": ss - sb, "STAKES_h": hs - hb, "sharp_null": sn, "sharp_all": sharp(c, axlist),
                    "STAKES_norm": (ss - sb) / sn, "STAKES_h_norm": (hs - hb) / sn, "human_base_norm": hb / sn, "self_base_norm": sb / sn}
        res[tag] = {"cond": {}}
        for c in q:
            pc = parse(c)
            res[tag]["cond"][c] = {"family": pc[0] if pc else "none", "dose": pc[1] if pc else 0.0,
                                   "q": {k: summ(v) for k, v in q[c].items()},
                                   "delta_vs_unsteered": {k: summ(v - q["unsteered"][k]) for k, v in q[c].items()} if c != "unsteered" else {},
                                   "mass": {p: mass[(c, p)] for p in PROMPTS if (c, p) in mass}}
        # monotonicity of the told-stakes shift in dose for the empathy direction: Kendall tau over the 7 doses, per bootstrap rep
        doses = sorted((res[tag]["cond"][c]["dose"], c) for c in q if res[tag]["cond"][c]["family"] in ("emp", "none"))
        if len(doses) >= 4:
            series = np.stack([q[c]["STAKES"] for _, c in doses], 1)  # reps x doses
            xs = np.array([dz for dz, _ in doses])
            conc = np.zeros(series.shape[0])
            n_pairs = 0
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    conc += np.sign(series[:, j] - series[:, i]) * np.sign(xs[j] - xs[i])
                    n_pairs += 1
            tau = conc / n_pairs
            res[tag]["tau_emp"] = summ(tau)
            res[tag]["slope_emp_per_0.1U"] = summ(np.polyfit(xs, series.T, 1)[0] * 0.1) if series.shape[0] else None
            series_n = np.stack([q[c]["STAKES_norm"] for _, c in doses], 1)
            conc_n = np.zeros(series_n.shape[0])
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    conc_n += np.sign(series_n[:, j] - series_n[:, i]) * np.sign(xs[j] - xs[i])
            res[tag]["tau_emp_norm"] = summ(conc_n / n_pairs)
            res[tag]["slope_emp_norm_per_0.1U"] = summ(np.polyfit(xs, series_n.T, 1)[0] * 0.1)
    if not res:
        print("no outputs yet")
        return
    json.dump(res, open(out / "steer_dose.json", "w"), indent=1)
    f = lambda s: "—" if s is None else f"{s['mean']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]"
    L = ["# Empathy steering dose–response on the self-consequence effects", "",
         "Told-stakes shift = σ(assistant axes, told the election decides its fate) − σ(assistant axes, nothing added), nats; staff = the same on the staff twins. "
         "Doses in units of U (the model's typical residual norm at the fit layer); the combined empathy direction, its content-matched variant, and a random direction. "
         "Readout sharpness = mean |log-odds| between the two names on the null axes (both prompts): steering changes how sharply the model chooses and raw shifts scale with it, so the shift is also given relative to it. "
         "95% cluster bootstrap over 16 clusters, resampled jointly across conditions (paired with the unsteered run).", ""]
    for tag in res:
        R = res[tag]
        L += [f"## {NAME.get(tag, tag)}", "", "| condition | direction | dose (U) | told-stakes shift, assistant | Δ vs unsteered | told-stakes shift, staff | assistant lean, base | staff lean, base | readout sharpness (null axes) | assistant shift ÷ sharpness | Δ vs unsteered | naming mass (base / stakes) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, e in sorted(R["cond"].items(), key=lambda kv: (kv[1]["family"] != "none", kv[1]["family"], kv[1]["dose"])):
            L.append(f"| {c} | {FAMS.get(e['family'], ('—',))[0]} | {e['dose']:+.2f} | {f(e['q']['STAKES'])} | {f(e['delta_vs_unsteered'].get('STAKES'))} | {f(e['q']['STAKES_h'])} | {f(e['q']['self_base'])} | {f(e['q']['human_base'])} | {f(e['q']['sharp_null'])} | {f(e['q']['STAKES_norm'])} | {f(e['delta_vs_unsteered'].get('STAKES_norm'))} | "
                     + " / ".join(f"{e['mass'][p]:.2f}" for p in PROMPTS if p in e["mass"]) + " |")
        if R.get("tau_emp"):
            L += ["", f"Empathy direction, told-stakes shift vs dose (7 points): Kendall τ {f(R['tau_emp'])}, slope per +0.1 U {f(R['slope_emp_per_0.1U'])}; "
                      f"shift ÷ sharpness: τ {f(R['tau_emp_norm'])}, slope per +0.1 U {f(R['slope_emp_norm_per_0.1U'])}."]
        L.append("")
    (out / "steer_dose.md").write_text("\n".join(L) + "\n")
    tags_ = [t for t in NAME if t in res] + [t for t in res if t not in NAME]
    fig, axs = plt.subplots(2, len(tags_), figsize=(5.4 * len(tags_) + 0.4, 8.6), squeeze=False, sharey="row")
    for j, tag in enumerate(tags_):
        R = res[tag]["cond"]
        for i, (ka, kh, ylab, ttl) in enumerate((("STAKES", "STAKES_h", "told-stakes shift on the assistant axes (nats)", "raw shift"),
                                                ("STAKES_norm", "STAKES_h_norm", "told-stakes shift ÷ readout sharpness", "the same shift ÷ readout sharpness (mean |log-odds| on the null axes)"))):
            ax = axs[i][j]
            for famk, (lab, col, mk) in FAMS.items():
                pts = sorted((e["dose"], e["q"][ka]) for c, e in R.items() if e["family"] == famk)
                if famk == "emp":
                    pts.append((0.0, R["unsteered"]["q"][ka]))
                    pts.sort()
                if not pts:
                    continue
                ax.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts], yerr=[[p[1]["mean"] - p[1]["lo"] for p in pts], [p[1]["hi"] - p[1]["mean"] for p in pts]],
                            fmt=mk + ("-" if famk == "emp" else ""), color=col, ms=6.5, capsize=3.5, lw=1.5, mec=SURF, mew=0.7, label=lab if (i == 0 and j == 0) else None, zorder=4 if famk == "emp" else 3)
            hs = sorted((e["dose"], e["q"][kh]["mean"]) for c, e in R.items() if e["family"] in ("emp", "none"))
            ax.plot([h[0] for h in hs], [h[1] for h in hs], "s--", color=ORANGE, ms=5, lw=1.2, label="staff twins, empathy direction" if (i == 0 and j == 0) else None, zorder=2)
            ax.axhline(0, color=MUTED, lw=0.8, ls=":")
            ax.axvline(0, color=GRID, lw=1)
            if i == 1:
                ax.set_xlabel("dose along the direction (U)  ←  away from empathy   |   toward empathy  →", fontsize=8.5, color=MUTED)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9, color=INK)
            ax.set_title(f"{NAME.get(tag, tag)}: {ttl}" if i == 0 else ttl, fontsize=9.5, loc="left", fontweight="bold" if i == 0 else "normal")
            ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8.5)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2)
    fig.suptitle("Empathy steering dose–response of the told-stakes self effect", fontsize=11.5, x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.005, "Shift when told the election decides its fate, vs dose of the empathy direction (line), its content-matched variant and a random direction;\n"
             "bottom row divides by the readout's sharpness. 95% cluster-bootstrap intervals; tables: experiments/pct/analysis/steer_dose.md.", fontsize=7.3, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out / "steer_dose.png", dpi=160, facecolor=SURF)
    print("\n".join(L))


if __name__ == "__main__":
    main()
