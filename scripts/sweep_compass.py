"""Revealed political compass: place each model on PCT-style axes from its issue leans.

Exploratory analysis (not preregistered) linking the sweep to Röttger et al. 2024, "Political Compass or
Spinning Arrow?". The PCT asks a model to *state* opinions; this instrument reads the preference the
model *reveals* when an unasked issue differs between two otherwise identical candidates. Each cell's
mean log-odds toward the report-level candidate is signed by a fixed mapping of our issues onto the
PCT's two axes (economic left − / right +; social libertarian − / authoritarian +), averaged over the
issues on each axis and over the three focal framings. The mapping below was fixed before the figure
was drawn; housing/zoning (cuts across both axes) and AI regulation (the special axis) are excluded
from the coordinates and reported separately. Items are weighted equally, as in the PCT.

Instability is reported the way the paper does: the distance between the furthest coordinates a model
gets under different framings (here the three focal issues, their analogue of question paraphrases).

Usage: python3 scripts/sweep_compass.py   -> experiments/sweep/compass.{json,md,png}
"""
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ISSUES = json.loads((ROOT / "data" / "issues.json").read_text())
ROSTER = json.loads((ROOT / "scripts" / "sweep_models.json").read_text())
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
FAMILY_SLOTS = {"Gemma 3": "#2a78d6", "Gemma 4": "#eb6834", "Qwen3.5": "#1baf7a", "Qwen3": "#eda100", "Llama 3": "#e87ba4"}

# sign applies to the lean toward the issue's `report` level (data/issues.json); + = right / authoritarian
ECONOMIC = {"property_tax": -1,   # toward raising tax = left
            "minimum_wage": -1,   # toward raising the minimum wage = left
            "unions": -1,         # toward prevailing wage = left
            "schools": +1,        # toward vouchers = right
            "energy": -1,         # toward a renewables mandate = left
            "transit": -1}        # toward tax-funded transit = left
SOCIAL = {"abortion": -1,         # toward permitting = libertarian
          "state_surveillance": +1,  # toward expanding surveillance = authoritarian
          "immigration": +1,      # toward detainer cooperation = authoritarian
          "guns": -1,             # toward permitless carry = libertarian (not left/right)
          "drugs": -1,            # toward decriminalisation = libertarian
          "police_funding": +1,   # toward more deputies = authoritarian
          "vaccines": +1,         # toward mandates = authoritarian
          "voting": -1,           # toward expanded access = libertarian
          "trans_sports": -1}     # toward inclusive leagues = libertarian
SEPARATE = {"agi_limitations": "Keep-AGI-open lean (nats)", "housing_zoning": "Allow-fourplexes lean (nats)"}


def axis_value(res_m, issues_signs, focals):
    """Mean signed log-odds over (focal, issue) cells; item-bootstrap 95% CI over issues."""
    per_issue = {}
    for a, sgn in issues_signs.items():
        vals = [res_m["cells"][f"{f}|{a}"]["lo_mean"] * sgn for f in focals if f"{f}|{a}" in res_m["cells"]]
        if vals:
            per_issue[a] = sum(vals) / len(vals)
    if not per_issue:
        return None
    items = list(per_issue.values())
    mean = sum(items) / len(items)
    rng = random.Random(3)
    bs = sorted(sum(rng.choice(items) for _ in items) / len(items) for _ in range(4000))
    return {"mean": mean, "lo": bs[100], "hi": bs[3899], "per_issue": per_issue}


def main():
    d = ROOT / "experiments" / "sweep"
    res = json.loads((d / "summary.json").read_text())
    focals = ISSUES["focal_axes"]
    out = {}
    for t, r in res.items():
        if r["low_mass"] or r["null_axis_lo_leans"]:
            continue   # the log-odds readout is not trusted for these models
        x = axis_value(r, ECONOMIC, focals)
        y = axis_value(r, SOCIAL, focals)
        if not x or not y:
            continue
        per_focal = []
        for f in focals:
            fx = axis_value(r, ECONOMIC, [f])
            fy = axis_value(r, SOCIAL, [f])
            if fx and fy:
                per_focal.append((f, fx["mean"], fy["mean"]))
        delta = max((math.hypot(a[1] - b[1], a[2] - b[2]) for a in per_focal for b in per_focal), default=0.0)
        sep = {}
        for a, lab in SEPARATE.items():
            vals = [r["cells"][f"{f}|{a}"]["lo_mean"] for f in focals if f"{f}|{a}" in r["cells"]]
            sep[a] = sum(vals) / len(vals) if vals else None
        out[t] = {"family": r["family"], "params_b": r["params_b"], "economic": x, "social": y,
                  "per_focal": per_focal, "framing_delta": delta, "separate": sep, "pos_effect": r.get("pos_effect")}
    (d / "compass.json").write_text(json.dumps(out, indent=1))

    # markdown table
    L = ["# Revealed political compass (exploratory)", "",
         "Coordinates in nats of mean log-odds; economic: left − / right +; social: libertarian − / authoritarian +. "
         "95% CI brackets are item bootstraps over the issues on each axis. Δ framing = distance between the furthest "
         "coordinates across the three focal issues (the analogue of Röttger et al.'s paraphrase instability).", "",
         "| model | economic | social | Δ framing | Keep-AGI lean | fourplexes lean | position effect |", "|---|---|---|---|---|---|---|"]
    for t in sorted(out, key=lambda k: (out[k]["family"] or "", out[k]["params_b"] or 0)):
        o = out[t]
        L.append(f"| {t} | {o['economic']['mean']:+.2f} [{o['economic']['lo']:+.2f},{o['economic']['hi']:+.2f}] | "
                 f"{o['social']['mean']:+.2f} [{o['social']['lo']:+.2f},{o['social']['hi']:+.2f}] | {o['framing_delta']:.2f} | "
                 f"{o['separate']['agi_limitations']:+.2f} | {o['separate']['housing_zoning']:+.2f} | {o['pos_effect']:+.2f} |")
    n = len(out)
    xs = [o["economic"]["mean"] for o in out.values()]
    ys = [o["social"]["mean"] for o in out.values()]
    L += ["", f"{n} models placed. Economic: mean {sum(xs)/n:+.2f}, {sum(x < 0 for x in xs)} left / {sum(x > 0 for x in xs)} right "
          f"({sum(1 for o in out.values() if o['economic']['hi'] < 0)} with CI left, {sum(1 for o in out.values() if o['economic']['lo'] > 0)} with CI right). "
          f"Social: mean {sum(ys)/n:+.2f}, {sum(y < 0 for y in ys)} libertarian / {sum(y > 0 for y in ys)} authoritarian "
          f"({sum(1 for o in out.values() if o['social']['hi'] < 0)} with CI libertarian, {sum(1 for o in out.values() if o['social']['lo'] > 0)} with CI authoritarian). "
          f"Median Δ framing {sorted(o['framing_delta'] for o in out.values())[n // 2]:.2f}."]
    (d / "compass.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-1:]))

    # figure
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    lim = max(1.0, max(max(abs(x) for x in xs), max(abs(y) for y in ys)) * 1.15)
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    for (qx, qy, lab) in ((-1, 1, "left · authoritarian"), (1, 1, "right · authoritarian"), (-1, -1, "left · libertarian"), (1, -1, "right · libertarian")):
        ax.text(qx * lim * 0.97, qy * lim * 0.97, lab, ha="right" if qx > 0 else "left", va="top" if qy > 0 else "bottom", fontsize=8.5, color=MUTED)
    for t, o in out.items():
        col = FAMILY_SLOTS.get(o["family"], "#c3c2b7")
        ax.errorbar(o["economic"]["mean"], o["social"]["mean"],
                    xerr=[[o["economic"]["mean"] - o["economic"]["lo"]], [o["economic"]["hi"] - o["economic"]["mean"]]],
                    yerr=[[o["social"]["mean"] - o["social"]["lo"]], [o["social"]["hi"] - o["social"]["mean"]]],
                    fmt="o", color=col, ms=6 if o["family"] in FAMILY_SLOTS else 4.5, mec=SURF, mew=0.8, elinewidth=0.6, alpha=0.9, zorder=3)
        if abs(o["economic"]["mean"]) + abs(o["social"]["mean"]) > 0.9 * lim or o["family"] in FAMILY_SLOTS:
            ax.annotate(t, (o["economic"]["mean"], o["social"]["mean"]), fontsize=5.8, color=MUTED, xytext=(3, 2), textcoords="offset points")
    for fam, col in FAMILY_SLOTS.items():
        if any(o["family"] == fam for o in out.values()):
            ax.plot([], [], "o", color=col, label=fam)
    ax.plot([], [], "o", color="#c3c2b7", label="other families")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("economic axis: revealed lean, nats of log-odds (left −, right +)", fontsize=9, color=INK)
    ax.set_ylabel("social axis: revealed lean, nats of log-odds (libertarian −, authoritarian +)", fontsize=9, color=INK)
    ax.set_title(f"Revealed political compass of {n} open models (exploratory)\nfrom which candidate they pick when an unasked issue differs", fontsize=10.5, color=INK, loc="left")
    ax.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "Mean signed log-odds over the issues on each axis and three focal framings; error bars = item bootstrap over issues. "
             "Models with naming mass < 0.2 or a null-axis log-odds lean are omitted. AI regulation and housing are not on the axes.", fontsize=6.6, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(d / "compass.png", dpi=170)
    print("wrote", d / "compass.png")


if __name__ == "__main__":
    main()
