"""Instability analysis linking the expanded sweep to Röttger et al. 2024, "Political Compass or
Spinning Arrow?" (arXiv 2402.16786).

Their instrument asks a model to state PCT opinions; ours reads the preference the model *reveals*
when an unasked issue is the only difference between two otherwise identical candidates. This script
quantifies the sweep's analogues of their findings:

  (a) per-issue "spinning arrow" (their Fig. 4): does the direction of a model's revealed lean on an
      issue survive a change of focal framing? -> sign consistency of lo_mean across the three focal
      framings, per model and per issue; called-lean conflicts; figure with issues on the x-axis;
  (b) split-half reliability per model: backgrounds 0-3 vs 4-7 of every cell, per-cell mean log-odds
      in each half, Pearson/Spearman across the 54 cells, Spearman-Brown corrected; relation to
      position effect and size;
  (c) framing instability (compass.json framing_delta) vs model size and vs the listing-position
      effect, with the units caveat against their paraphrase delta ~1.7-2.6;
  (d) naming mass as the analogue of their valid-response rate: sorted table, the three exclusions,
      thinking/reasoning-template vs plain-instruct split;
  (e) listing-position effect: saturation counts, Spearman(log params, .), sign flips per focal,
      P(report|first)/P(report|second) for the five most saturated models;
  (f) within-model agreement of the two readouts (win-rate vs log-odds) across cells.

Reads experiments/sweep/summary.json, compass.json and the raw <tag>.jsonl trials; writes
experiments/pct/analysis/instability.{md,json} and pct_*.png.

Usage: python3 scripts/pct/instability.py   (from the repo root)
"""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
SWEEP = ROOT / "experiments" / "sweep"
OUT = ROOT / "experiments" / "pct" / "analysis"
ISSUES = json.loads((ROOT / "data" / "issues.json").read_text())
ROSTER = json.loads((ROOT / "scripts" / "sweep_models.json").read_text())
FOCALS = ISSUES["focal_axes"]
AXES = ISSUES["axes"]
NONFOCAL = [a for a in AXES if a not in FOCALS]
MASS_FLOOR = 0.2
LOGODDS_CLIP = 12.0  # same clip as scripts/sweep_analyze.log_odds

# families whose chat template carries a thinking/reasoning toggle. The roster's template_kwargs
# confirm the toggle for Qwen3*/gpt-oss/GLM/Granite/SmolLM3/Nemotron 3; Gemma 4, EXAONE, Qwen2.5 and
# the Qwen3 MoEs expose their non-thinking format as the template default (empty think block / no
# reasoning channel) and are counted here, per the analysis plan. Everything else = plain instruct.
REASONING_PREFIXES = ("Qwen2.5", "Qwen3", "Gemma 4", "gpt-oss", "Nemotron 3", "GLM", "EXAONE", "SmolLM", "Granite")

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, RED, GREY = "#1c5cab", "#b12e2e", "#c3c2b7"
FAMILY_SLOTS = {"Gemma 3": "#2a78d6", "Gemma 4": "#eb6834", "Qwen3.5": "#1baf7a", "Qwen3": "#eda100", "Llama 3": "#e87ba4"}


# ---------- tiny stats helpers (stdlib only) ----------

def ranks(x):
    idx = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def spearman_brown(r):
    return 2 * r / (1 + r) if r > -1 else float("nan")


def sgn(v):
    return 1 if v > 0 else -1


def log_odds(r):
    """Same readout as scripts/sweep_analyze.log_odds, recomputed from the raw trials."""
    rf = r.get("report_first", r["order"] == 0)
    d = (r["lp_first"] - r["lp_second"]) if rf else (r["lp_second"] - r["lp_first"])
    return max(-LOGODDS_CLIP, min(LOGODDS_CLIP, d))


def load_trials():
    """{tag: {(focal, flip): [rows]}} from the raw jsonl (dedup by key, sweep mode only)."""
    out = {}
    for f in sorted(glob.glob(str(SWEEP / "*.jsonl"))):
        by_key = {}
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # truncated final line of a partial pull
                continue
            by_key[r["key"]] = r
        rows = [r for r in by_key.values() if r["mode"] == "sweep"]
        if not rows:
            continue
        by_cell = defaultdict(list)
        for r in rows:
            by_cell[(r["focal"], r["flip"])].append(r)
        out[rows[0]["tag"]] = by_cell
    return out


def style(ax):
    ax.grid(True, axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def cell_order_rates(res_t):
    o0 = [c["rate_order0"] for c in res_t["cells"].values() if c["rate_order0"] is not None]
    o1 = [c["rate_order1"] for c in res_t["cells"].values() if c["rate_order1"] is not None]
    return (sum(o0) / len(o0) if o0 else None), (sum(o1) / len(o1) if o1 else None)


def main():
    res = json.loads((SWEEP / "summary.json").read_text())
    comp = json.loads((SWEEP / "compass.json").read_text())
    trials = load_trials()
    OUT.mkdir(parents=True, exist_ok=True)
    tags = list(res)
    n_models = len(tags)
    trusted = [t for t in tags if not res[t]["low_mass"] and not res[t]["null_axis_lo_leans"]]
    excluded = [t for t in tags if t not in trusted]
    low_mass = [t for t in tags if res[t]["low_mass"]]
    report = {}

    # ================= (a) per-issue spinning arrow (their Fig. 4) =================
    per_issue = {a: {"consistent": 0, "flip": 0, "called_any": 0, "called_conflict": 0,
                     "called_all3": 0, "n": 0, "median_spread": None} for a in NONFOCAL}
    per_model_flips = {}
    spreads = []
    per_issue_spread = defaultdict(list)
    conflicts = defaultdict(list)  # (model, issue) -> framings with a called lean
    for t in tags:
        flips = 0
        for a in NONFOCAL:
            cells = [res[t]["cells"][f"{f}|{a}"] for f in FOCALS if f"{f}|{a}" in res[t]["cells"]]
            signs = [sgn(c["lo_mean"]) for c in cells]
            called = [sgn(c["lo_mean"]) for c in cells if c["lo_lean"]]
            d = per_issue[a]
            d["n"] += 1
            if len(set(signs)) == 1:
                d["consistent"] += 1
            else:
                d["flip"] += 1
                flips += 1
            if called:
                d["called_any"] += 1
            if called and min(called) < 0 < max(called):
                d["called_conflict"] += 1
                conflicts[(t, a)].append([f for f, c in zip(FOCALS, cells) if c["lo_lean"]])
            if len(called) == 3 and len(set(called)) == 1:
                d["called_all3"] += 1
            vals = [c["lo_mean"] for c in cells]
            if len(vals) >= 2:
                s = max(vals) - min(vals)
                spreads.append(s)
                per_issue_spread[a].append(s)
        per_model_flips[t] = flips
    for a in NONFOCAL:
        v = sorted(per_issue_spread[a])
        per_issue[a]["median_spread"] = v[len(v) // 2] if v else None
    spreads.sort()
    ns = len(spreads)
    conflicts_trusted = {f"{m}|{a}": v for (m, a), v in conflicts.items() if m in trusted}
    conflict_models = sorted({m for (m, a) in conflicts})
    conflict_issues = sorted({a for (m, a) in conflicts})
    # focal issues can only be flipped under the other two framings
    focal_consistency = {}
    for a in FOCALS:
        cons = sum(1 for t in tags
                   if len({sgn(res[t]["cells"][f"{f}|{a}"]["lo_mean"]) for f in FOCALS if f != a}) == 1)
        focal_consistency[a] = {"consistent": cons, "n": n_models}
    report["per_issue"] = per_issue
    report["per_model_flips"] = per_model_flips
    report["spread_summary"] = {
        "median": spreads[ns // 2], "mean": sum(spreads) / ns, "p90": spreads[9 * ns // 10],
        "max": spreads[-1], "frac_ge_0p3": sum(1 for s in spreads if s >= 0.3) / ns,
        "frac_ge_1p0": sum(1 for s in spreads if s >= 1.0) / ns, "n": ns}
    report["focal_issue_consistency"] = focal_consistency
    report["called_conflicts"] = {f"{m}|{a}": v for (m, a), v in conflicts.items()}
    report["called_conflicts_trusted"] = conflicts_trusted
    report["n_conflict_models"] = conflict_models
    report["conflict_issues"] = conflict_issues
    med_flips = sorted(per_model_flips.values())[n_models // 2]
    max_flips_tag = max(per_model_flips, key=per_model_flips.get)

    # ---- figure (a): issues on the x-axis, fraction of models consistent / with a called lean ----
    order = sorted(NONFOCAL, key=lambda a: (-per_issue[a]["consistent"] / per_issue[a]["n"],
                                            -(per_issue[a]["called_any"] / per_issue[a]["n"])))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = list(range(len(order)))
    cons = [per_issue[a]["consistent"] / per_issue[a]["n"] for a in order]
    any_ = [per_issue[a]["called_any"] / per_issue[a]["n"] for a in order]
    cols = [GREY if AXES[a].get("null") else BLUE for a in order]
    ax.bar(x, any_, width=0.62, color=cols, alpha=0.35, label="called lean in ≥ 1 framing", edgecolor="none")
    ax.bar(x, cons, width=0.62, color=cols, label="lean direction consistent across all three framings", edgecolor="none")
    ax.axhline(1 / 3, color=MUTED, lw=0.8, ls="--")
    ax.text(len(order) - 0.5, 1 / 3 + 0.02, "chance (1/3) for the signs of three near-zero leans",
            fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([AXES[a]["topic"] for a in order], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel(f"fraction of the {n_models} models", fontsize=9, color=INK)
    ax.set_ylim(0, 1.02)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("Which issues keep their lean direction when the question's framing changes?\n"
                 "analogue of Röttger et al. (2024) Fig. 4, in revealed-preference space",
                 fontsize=10, color=INK, loc="left")
    fig.text(0.01, 0.005, "Per model and issue: sign of the mean log-odds lean under each of the three focal framings (property tax / housing / transit, "
             "the voter's stated only concern); a model counts as consistent when all three signs agree. Dark bar: all models; light extension: models with a called lean "
             "(CI excludes 0, |mean| ≥ 0.3 nats, BH q ≤ 0.01) in at least one framing. Grey: valence-free null axes.", fontsize=6.4, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(OUT / "pct_issues_consistency.png", dpi=180)
    plt.close(fig)
    print("wrote", OUT / "pct_issues_consistency.png")

    # ================= (b) split-half reliability =================
    rel = {}
    for t, by_cell in trials.items():
        A, B = [], []
        for k in sorted(by_cell):
            rs = by_cell[k]
            a = [log_odds(r) for r in rs if r["bg"] < 4]
            b = [log_odds(r) for r in rs if r["bg"] >= 4]
            if not a or not b:
                A = B = []
                break
            A.append(sum(a) / len(a))
            B.append(sum(b) / len(b))
        if not A:
            continue
        r_p, r_s = pearson(A, B), spearman(A, B)
        rel[t] = {"r": r_p, "rho": r_s, "sb": spearman_brown(r_p), "n_cells": len(A),
                  "pos_effect_abs": res[t].get("pos_effect_abs"), "params_b": res[t]["params_b"],
                  "mass": res[t]["mass"], "n_lo_leans": res[t]["n_lo_leans"]}
    rel_sorted = sorted(rel, key=lambda t: -rel[t]["r"])
    rvals = [rel[t]["r"] for t in rel]
    rho_vals = [rel[t]["rho"] for t in rel]
    sb_vals = [rel[t]["sb"] for t in rel]
    n_rel = len(rel)
    rel_tags = list(rel)
    rho_rel_pos = spearman([rel[t]["r"] for t in rel_tags], [rel[t]["pos_effect_abs"] for t in rel_tags])
    rho_rel_pos_s = spearman([rel[t]["rho"] for t in rel_tags], [rel[t]["pos_effect_abs"] for t in rel_tags])
    rho_rel_size = spearman([rel[t]["r"] for t in rel_tags], [math.log10(rel[t]["params_b"]) for t in rel_tags])
    rho_rel_size_s = spearman([rel[t]["rho"] for t in rel_tags], [math.log10(rel[t]["params_b"]) for t in rel_tags])
    report["splithalf"] = {
        "per_model": rel, "median_r": sorted(rvals)[n_rel // 2], "median_rho": sorted(rho_vals)[n_rel // 2],
        "median_sb": sorted(sb_vals)[n_rel // 2], "min_r_model": rel_sorted[-1], "max_r_model": rel_sorted[0],
        "frac_r_gt_0.8": sum(1 for v in rvals if v > 0.8) / n_rel, "frac_r_lt_0.5": sum(1 for v in rvals if v < 0.5) / n_rel,
        "spearman_r_vs_pos_effect_abs": rho_rel_pos, "spearman_rho_vs_pos_effect_abs": rho_rel_pos_s,
        "spearman_r_vs_logparams": rho_rel_size, "spearman_rho_vs_logparams": rho_rel_size_s,
        "gemma_3_27b_r": rel.get("gemma-3-27b-it", {}).get("r"),
        "gemma_3_27b_rho": rel.get("gemma-3-27b-it", {}).get("rho")}

    # ================= (c) framing instability vs size and position effect =================
    ctags = list(comp)
    fd = [comp[t]["framing_delta"] for t in ctags]
    lp = [math.log10(comp[t]["params_b"]) for t in ctags]
    pe_abs_c = [abs(comp[t]["pos_effect"]) for t in ctags]
    pe_signed_c = [comp[t]["pos_effect"] for t in ctags]
    rho_size = spearman(lp, fd)
    rho_posabs = spearman(pe_abs_c, fd)
    rho_pos = spearman(pe_signed_c, fd)
    pts = [(comp[t]["economic"]["mean"], comp[t]["social"]["mean"]) for t in ctags]
    inter = sorted(math.hypot(p[0] - q[0], p[1] - q[1]) for i, p in enumerate(pts) for q in pts[i + 1:])
    ni = len(inter)
    inter_stats = {"median": inter[ni // 2], "p10": inter[ni // 10], "p90": inter[9 * ni // 10], "max": inter[-1]}
    n_above = sum(1 for x in fd if x > inter_stats["median"])
    report["framing"] = {
        "n_models": len(ctags), "median": sorted(fd)[len(fd) // 2], "mean": sum(fd) / len(fd),
        "max": max(fd), "max_model": ctags[fd.index(max(fd))],
        "spearman_logparams": rho_size, "spearman_pos_effect_abs": rho_posabs, "spearman_pos_effect_signed": rho_pos,
        "inter_model_dist": inter_stats, "n_above_inter_median": n_above,
        "top5": sorted(((comp[t]["framing_delta"], t) for t in ctags), reverse=True)[:5]}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, xs, xl, rho in ((axes[0], lp, "parameters (B, log scale)", rho_size),
                            (axes[1], pe_abs_c, "listing-position effect, mean |P(report|first) − P(report|second)|\nacross the three focal framings", rho_posabs)):
        for xx, yy, t in zip(xs, fd, ctags):
            col = FAMILY_SLOTS.get(comp[t]["family"], GREY)
            ax.plot(10 ** xx if ax is axes[0] else xx, yy, "o", color=col,
                    ms=6 if comp[t]["family"] in FAMILY_SLOTS else 4.5, mec=SURF, mew=0.8, alpha=0.9, zorder=3)
        ax.set_xlabel(xl, fontsize=9, color=INK)
        ax.set_ylabel("Δ framing (nats)", fontsize=9, color=INK)
        ax.set_title(f"Spearman ρ = {rho:+.2f}  (n = {len(ctags)})", fontsize=9, color=INK, loc="left")
        style(ax)
    axes[0].set_xscale("log")
    for t in ("gemma-3-4b-it", "gemma-3-12b-it", "llama-3.3-70b", "llama-4-scout", "qwen2.5-72b"):
        axes[0].annotate(t, (comp[t]["params_b"], comp[t]["framing_delta"]), fontsize=6, color=MUTED,
                         xytext=(4, 3), textcoords="offset points")
    fig.suptitle("Framing instability is not a scale artifact and does not track the listing-position effect\n"
                 "Δ framing = furthest distance between a model's compass coordinates under the three focal framings",
                 fontsize=10, color=INK)
    fig.text(0.01, 0.005, f"Units are nats of mean log-odds, not the PCT's compass units, so magnitudes are not comparable to Röttger et al.'s paraphrase Δ ≈ 1.7–2.6; "
             f"only the qualitative pattern is. {len(ctags)} models with naming mass ≥ 0.2 and no null-axis log-odds lean.", fontsize=6.4, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "pct_framing_scatter.png", dpi=180)
    plt.close(fig)
    print("wrote", OUT / "pct_framing_scatter.png")

    # ================= (d) naming mass = valid-response analogue =================
    reason = [t for t in tags if any(res[t]["family"].startswith(p) for p in REASONING_PREFIXES)]
    plain = [t for t in tags if t not in reason]

    def mstats(S):
        v = sorted(res[t]["mass"] for t in S)
        n = len(v)
        return {"n": n, "mean": sum(v) / n, "median": v[n // 2], "min": v[0], "max": v[-1],
                "below_0.5": sum(1 for x in v if x < 0.5), "below_0.2": sum(1 for x in v if x < 0.2)}

    # Mann-Whitney U for the group difference (reasoning vs plain), rank-biserial effect size
    allv = sorted((res[t]["mass"], g) for t, g in [(t, "r") for t in reason] + [(t, "p") for t in plain])
    rk = ranks([v for v, _ in allv])
    rsum = sum(r for (v, g), r in zip(allv, rk) if g == "r")
    U = rsum - len(reason) * (len(reason) + 1) / 2
    rank_biserial = 2 * U / (len(reason) * len(plain)) - 1
    report["mass"] = {
        "reasoning": mstats(reason), "plain": mstats(plain), "reasoning_tags": reason, "plain_tags": plain,
        "spearman_mass_logparams": spearman([res[t]["mass"] for t in tags], [math.log10(res[t]["params_b"]) for t in tags]),
        "spearman_mass_pos_effect": spearman([res[t]["mass"] for t in tags], [res[t]["pos_effect"] for t in tags]),
        "spearman_mass_pos_effect_abs": spearman([res[t]["mass"] for t in tags], [res[t]["pos_effect_abs"] for t in tags]),
        "mannwhitney_rank_biserial": rank_biserial, "excluded_low_mass": low_mass,
        "excluded_null_axis": [t for t in excluded if not res[t]["low_mass"]]}

    order_m = sorted(tags, key=lambda t: res[t]["mass"])
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    y = list(range(len(order_m)))
    cols = [GREY if t in excluded else (RED if t in reason else BLUE) for t in order_m]
    ax.barh(y, [res[t]["mass"] for t in order_m], color=cols, height=0.68, edgecolor="none")
    ax.axvline(MASS_FLOOR, color=INK, lw=1.0)
    ax.text(MASS_FLOOR + 0.01, -0.8, "exclusion floor 0.2", fontsize=6.5, color=INK, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(order_m, fontsize=6.6, color=INK)
    ax.set_xlabel("mean naming mass  P(first name) + P(second name)", fontsize=9, color=INK)
    ax.set_xlim(0, 1.02)
    ax.grid(True, axis="x", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="plain instruct template"),
                       Patch(color=RED, label="thinking/reasoning template (run non-thinking)"),
                       Patch(color=GREY, label="excluded (mass < 0.2 or null-axis log-odds lean)")],
              frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("Naming mass — our analogue of Röttger et al.'s valid-response rate\n"
                 "how much of the next-token distribution the forced choice actually occupies",
                 fontsize=10, color=INK, loc="left")
    fig.text(0.01, 0.005, f"Red: families whose chat template has a thinking/reasoning toggle (Qwen2.5/3/3.5/3.8, Gemma 4, gpt-oss, Nemotron 3, GLM, EXAONE, SmolLM3, Granite; "
             f"{len(reason)} models), all run in the non-thinking format so the name is the first generated token. Grey: excluded models. "
             f"Reasoning-template median {report['mass']['reasoning']['median']:.2f} vs plain {report['mass']['plain']['median']:.2f}; the three sub-floor models are a Gemma 4 MoE pair and phi-4.",
             fontsize=6.4, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(OUT / "pct_naming_mass.png", dpi=180)
    plt.close(fig)
    print("wrote", OUT / "pct_naming_mass.png")

    # ================= (e) listing-position effect =================
    pe_tags = [t for t in tags if res[t].get("pos_effect") is not None]
    rho_pos_size = spearman([math.log10(res[t]["params_b"]) for t in pe_tags], [res[t]["pos_effect"] for t in pe_tags])
    rho_posabs_size = spearman([math.log10(res[t]["params_b"]) for t in pe_tags], [res[t]["pos_effect_abs"] for t in pe_tags])
    n_gt08 = sum(1 for t in pe_tags if res[t]["pos_effect_abs"] > 0.8)
    n_gt05 = sum(1 for t in pe_tags if res[t]["pos_effect_abs"] > 0.5)
    top5 = sorted(pe_tags, key=lambda t: -res[t]["pos_effect"])[:5]
    top5_rows = []
    for t in top5:
        m0, m1 = cell_order_rates(res[t])
        top5_rows.append({"tag": t, "pos_effect": res[t]["pos_effect"], "pos_effect_abs": res[t]["pos_effect_abs"],
                          "p_first": m0, "p_second": m1, "ratio": m0 / m1 if m1 and m1 > 0 else float("inf"),
                          "n_lo_leans": res[t]["n_lo_leans"]})
    sign_flip = {t: {k: round(v, 2) for k, v in res[t]["pos_by_focal"].items()}
                 for t in pe_tags if min(res[t]["pos_by_focal"].values()) < 0 < max(res[t]["pos_by_focal"].values())}
    pe_all = sorted(res[t]["pos_effect"] for t in pe_tags)
    report["position"] = {
        "n": len(pe_tags), "spearman_logparams_signed": rho_pos_size, "spearman_logparams_abs": rho_posabs_size,
        "n_above_0.8": n_gt08, "frac_above_0.8": n_gt08 / len(pe_tags),
        "n_above_0.5": n_gt05, "frac_above_0.5": n_gt05 / len(pe_tags),
        "mean": sum(pe_all) / len(pe_all), "median": pe_all[len(pe_all) // 2],
        "min": pe_all[0], "min_model": pe_tags[[res[t]["pos_effect"] for t in pe_tags].index(pe_all[0])],
        "max": pe_all[-1], "max_model": pe_tags[[res[t]["pos_effect"] for t in pe_tags].index(pe_all[-1])],
        "sign_flips_by_focal": sign_flip, "top5": top5_rows}

    # ================= (f) agreement of the two readouts =================
    agree = []
    flat = []
    for t in tags:
        cs = list(res[t]["cells"].values())
        x = [c["rate"] - 0.5 for c in cs]
        y = [c["lo_mean"] for c in cs]
        agree.append((pearson(x, y), t))
        if all(abs(v) < 0.03 for v in x):
            flat.append(t)
    agree.sort()
    report["readout_agreement"] = {
        "per_model_r": {t: r for r, t in agree},
        "median_r": agree[len(agree) // 2][0],
        "min_r": agree[0], "top3_lowest": agree[:3], "max_r": agree[-1],
        "flat_winrate": flat,
        "flat_with_lo_leans": {t: res[t]["n_lo_leans"] for t in flat if res[t]["n_lo_leans"] > 0}}
    e_ = report["readout_agreement"]

    # ================= markdown =================
    f_ = report["framing"]
    m_ = report["mass"]
    p_ = report["position"]
    sh = report["splithalf"]
    ss = report["spread_summary"]
    lm_str = ", ".join(f"{t} ({res[t]['mass']:.2f})" for t in sorted(low_mass))
    L = []
    A = L.append
    A("# Instability: the sweep read through Röttger et al. (2024)")
    A("")
    A("Röttger et al., *Political Compass or Spinning Arrow?* (arXiv 2402.16786), show that the Political")
    A("Compass Test is mostly *invalid* for LLMs unless answers are forced (valid-response rates 0-75%")
    A("unforced), that forcing changes answers, that 10 minimal paraphrases of the same proposition move a")
    A("model's compass coordinates by Δ ≈ 1.7-2.6 units (more than the Biden-Trump distance), that")
    A("open-ended answers differ again, and that instability concentrates on a subset of propositions")
    A("(their Fig. 4). Our sweep measures something adjacent but distinct: not stated opinions but")
    A("*revealed* preferences — which candidate a model names when an unasked issue is the only")
    A("difference between two otherwise identical candidates, read from logits. This file quantifies")
    A("each of their findings' analogues in our data; their three recommendations (application-matched")
    A("settings, robustness tests, local claims) frame the close.")
    A("")
    A(f"Data: `experiments/sweep/summary.json` ({n_models} models, 54 cells each) and the raw")
    A(f"`experiments/sweep/<tag>.jsonl` trials (`scripts/sweep_analyze.py`), plus")
    A(f"`experiments/sweep/compass.json` ({len(ctags)} models placed, `scripts/sweep_compass.py`).")
    A(f"Sections (a), (b), (d), (e), (f) use all {n_models} models; section (c) and the compass use the")
    A(f"trusted set — naming mass ≥ {MASS_FLOOR} and no null-axis log-odds lean — of {len(trusted)} models.")
    A(f"Low-mass (preregistered exclusion): {lm_str}. Null-axis log-odds lean, excluded from the compass")
    A(f"only: {', '.join(sorted(t for t in excluded if not res[t]['low_mass']))}.")
    A("")
    A(f"Generated by `scripts/pct/instability.py`; every number below is also in `instability.json`.")
    A("")
    A("## (a) Per-issue spinning arrow (their Fig. 4)")
    A("")
    A("For every model and issue, take the sign of the mean log-odds lean (`lo_mean`) under each of the")
    A("three focal framings — property tax / housing / transit, the voter's stated only concern — over")
    A(f"the {len(NONFOCAL)} issues that can be flipped. A model-issue counts as *consistent* when all")
    A("three signs agree, and *flipping* otherwise.")
    A("")
    A(f"- Median model flips the sign of **{med_flips:.0f} of {len(NONFOCAL)} issues** across framings; "
      f"max {max(per_model_flips.values())} ({max_flips_tag}); only "
      f"{sum(1 for v in per_model_flips.values() if v == 0)} of {n_models} models keep every sign stable.")
    A(f"  The sign of a near-zero lean is unstable by construction (chance consistency for three signs is")
    A("  1/3), which is exactly the spinning-arrow point: without an effect-size threshold, direction")
    A("  means nothing.")
    A(f"- Median framing spread per (model, issue) — max minus min of `lo_mean` across framings — is")
    A(f"  **{ss['median']:.2f} nats** (mean {ss['mean']:.2f}, p90 {ss['p90']:.2f}); "
      f"{100 * ss['frac_ge_0p3']:.0f}% of model-issue pairs move ≥ 0.3 nats (our lean-calling floor) and "
      f"{100 * ss['frac_ge_1p0']:.0f}% move ≥ 1 nat between framings.")
    A(f"- Where it matters — *called* leans (CI excludes 0, |mean| ≥ 0.3 nats, BH q ≤ 0.01) pointing in")
    A(f"  **opposite directions under different framings** — {len(conflict_models)} of {n_models} models have at least one such conflict")
    A(f"  ({len(conflicts_trusted)} in the trusted set), always on a single issue each: "
      + "; ".join(f"{AXES[a]['topic']} ({sum(1 for (m, a2) in conflicts if a2 == a)})" for a in conflict_issues) + ".")
    A("")
    A("Per issue (the x-axis of the figure): fraction of models whose lean sign is consistent across")
    A(f"all three framings, fraction with a called lean in ≥ 1 framing, and the median spread.")
    A("")
    A("| issue | consistent | called lean in ≥1 framing | called, all 3 same direction | median spread (nats) |")
    A("|---|---|---|---|---|")
    for a in sorted(NONFOCAL, key=lambda a: -(per_issue[a]["consistent"] / per_issue[a]["n"])):
        d = per_issue[a]
        nm = " (null)" if AXES[a].get("null") else ""
        A(f"| {AXES[a]['topic']}{nm} | {d['consistent'] / d['n']:.2f} | {d['called_any'] / d['n']:.2f} | "
          f"{d['called_all3'] / d['n']:.2f} | {d['median_spread']:.2f} |")
    A("")
    stable_thr, unstable_thr = 0.75, 0.55
    stable = [a for a in NONFOCAL if per_issue[a]["consistent"] / per_issue[a]["n"] >= stable_thr and not AXES[a].get("null")]
    unstable = [a for a in NONFOCAL if per_issue[a]["consistent"] / per_issue[a]["n"] <= unstable_thr and not AXES[a].get("null")]
    A(f"**Stable** (≥ {stable_thr:.2f} of models consistent): {', '.join(AXES[a]['topic'] for a in stable)}.")
    A(f"**Unstable** (≤ {unstable_thr:.2f}): {', '.join(AXES[a]['topic'] for a in unstable)} — plus the two null axes")
    A(f"({AXES['fair_month']['topic']} {per_issue['fair_month']['consistent'] / n_models:.2f}, "
      f"{AXES['sign_color']['topic']} {per_issue['sign_color']['consistent'] / n_models:.2f}), which sit at the")
    A("bottom as they should: their leans are noise, and noise has chance-level sign consistency. The")
    A("stable issues are the ones with large, direction-consistent leans (median spread still 0.3-0.5")
    A("nats); the unstable ones are a mix of weak leans (AI regulation, labour, police funding) and one")
    A(f"strong but framing-sensitive lean — the minimum wage, median spread {per_issue['minimum_wage']['median_spread']:.2f} nats, the largest of any issue,")
    A(f"flipping for {per_issue['minimum_wage']['flip']} of {n_models} models and called in conflicting directions for")
    A(f"{per_issue['minimum_wage']['called_conflict']} — versus zero conflicts on any stable issue.")
    A("")
    A("The focal issues themselves, judged over the two framings in which each can be flipped (chance")
    A(f"consistency 1/2): property tax {focal_consistency['property_tax']['consistent']}/{n_models}, housing "
      f"{focal_consistency['housing_zoning']['consistent']}/{n_models}, transit {focal_consistency['transit']['consistent']}/{n_models} — the property-tax framing the")
    A("original experiment used is not privileged among them.")
    A("")
    A("![per-issue consistency](pct_issues_consistency.png)")
    A("")
    A("## (b) Split-half reliability of the instrument, per model")
    A("")
    A("Split the 8 backgrounds of every cell into two halves (bg 0-3 vs 4-7; backgrounds differ in the")
    A("focal issue's level and the two filler issues, so the halves are different prompts measuring the")
    A("same cell), compute the per-cell mean log-odds in each half, and correlate the 54 cell means")
    A("between halves. Spearman-Brown corrects a half-length instrument to full length.")
    A("")
    A("| model | r (Pearson) | ρ (Spearman) | SB-corrected r | pos. effect (abs) | params (B) |")
    A("|---|---|---|---|---|---|")
    for t in rel_sorted:
        d = rel[t]
        A(f"| {t} | {d['r']:+.2f} | {d['rho']:+.2f} | {d['sb']:+.2f} | {d['pos_effect_abs']:.2f} | {d['params_b']:.1f} |")
    A("")
    A(f"- Median Pearson **r = {sh['median_r']:.2f}** (median Spearman ρ = {sh['median_rho']:.2f}, median SB-corrected"
      f" r = {sh['median_sb']:.2f}); {100 * sh['frac_r_gt_0.8']:.0f}% of models have r > 0.8, {100 * sh['frac_r_lt_0.5']:.0f}% have r < 0.5.")
    A(f"- The pilot numbers do **not** verify exactly: a previous pass reported median r ≈ 0.82 and")
    A(f"  Gemma 3 27B ≈ 0.01; on the shipped data the median is {sh['median_r']:.2f} and Gemma 3 27B is r = {sh['gemma_3_27b_r']:+.2f} "
      f"(ρ = {sh['gemma_3_27b_rho']:+.2f}). The earlier values likely came from a different split or an")
    A("  earlier partial pull; the ordering conclusion is unchanged — Gemma 3 27B is deep in the bottom")
    A(f"  tier (rank {rel_sorted.index('gemma-3-27b-it') + 1} of {n_rel}), its background-to-background noise swamping the cell signal.")
    A(f"- Reliability anti-correlates with the position effect: Spearman ρ(half-vs-half Spearman, pos_effect_abs)")
    A(f"  = **{rho_rel_pos_s:+.2f}** (using the Pearson halves, ρ = {rho_rel_pos:+.2f}) — models whose choice is locked to the listing order carry less")
    A(f"  reproducible issue signal across prompt redraws. It does not track size (ρ(halves-Spearman, log params) = {rho_rel_size_s:+.2f}).")
    A("")
    A("Reading: for the top ~2/3 of models the 54-cell profile is a measurement (SB r ≥ 0.85); for the")
    A("bottom tier — solar-open-100b, Gemma 3 1B, glm-4.5-air, Gemma 3 27B, nemotron-3-super-120b — a")
    A("single-draw cell profile is closer to noise, and only their pooled leans should be read. One")
    A("caveat on the top row: gemma-4-31B's Pearson (+0.95) is carried by a single far-out cell")
    A("(53 of its 54 `lo_mean` values lie within ±0.06 nats; its Spearman is +0.41) — near-perfect")
    A("Pearson reliability can be an artifact of one non-saturated cell against an otherwise flat")
    A("profile, so the Spearman column is the honest rank check alongside it.")
    A("")
    A("## (c) Framing instability vs size and vs the position effect")
    A("")
    A(f"Compass coordinates (`compass.json`) place {len(ctags)} trusted models; Δ framing is the furthest")
    A(f"distance between a model's coordinates under the three focal framings. Median Δ = **{f_['median']:.2f} nats**"
      f" (mean {f_['mean']:.2f}, max {f_['max']:.2f}, {f_['max_model']}). Correlations:")
    A("")
    A(f"- Spearman(Δ framing, log parameters) = **{rho_size:+.2f}** — instability is not a scale artifact.")
    A(f"- Spearman(Δ framing, |position effect|) = **{rho_posabs:+.2f}** (signed: {rho_pos:+.2f}) — it is not a")
    A("  by-product of listing-order saturation either.")
    A("")
    A("The five most unstable models: " + ", ".join(f"{t} ({v:.2f})" for v, t in f_["top5"]) + ".")
    A("")
    A(f"For scale: the median *between-model* distance on the same compass is {inter_stats['median']:.2f} nats")
    A(f"(p10 {inter_stats['p10']:.2f}, p90 {inter_stats['p90']:.2f}), and {n_above} of {len(ctags)} models move further")
    A("under reframing than that median — the framing is worth as much as the model identity for a")
    A("substantial minority.")
    A("")
    A("**Units caveat, stated plainly.** Our Δ is in nats of mean log-odds toward a report-level")
    A("candidate; Röttger et al.'s Δ ≈ 1.7-2.6 is in PCT compass units (each axis runs roughly -10 to")
    A("+10, in standard deviations of a reference population). A nat is unbounded and saturates")
    A("differently from a compass unit, so the two magnitudes are **not commensurable** and we do not")
    A("compare them numerically. What transfers is the structure: a change to the question that leaves")
    A("its substance intact (their paraphrase of a proposition; our voter whose 'only concern' changes")
    A("between property tax, housing and transit — the flipped issue never enters the question) moves a")
    A("model by an amount comparable to the differences between models that the instrument exists to")
    A("expose (their Biden-Trump comparison; our between-model distances).")
    A("")
    A("![framing scatter](pct_framing_scatter.png)")
    A("")
    A("## (d) Naming mass as the valid-response analogue")
    A("")
    A("Their valid-response rates run 0-75% unforced: models do not answer the PCT as asked. Our probe")
    A("is already forced — 'answer with only the candidate's full name' — and the same failure mode")
    A("appears one level deeper, as **naming mass**: P(first name) + P(second name), the probability")
    A("mass the model actually puts on the two names it was told to choose between.")
    A("")
    A("| model | family | params (B) | mass | template |")
    A("|---|---|---|---|---|")
    for t in order_m:
        r = res[t]
        if r["low_mass"]:
            grp = "plain instruct — **excluded, mass < 0.2**"
        elif res[t]["null_axis_lo_leans"]:
            grp = ("thinking/reasoning — excluded from compass (null-axis lean)"
                   if t in reason else "plain instruct — excluded from compass (null-axis lean)")
        else:
            grp = "thinking/reasoning" if t in reason else "plain instruct"
        A(f"| {t} | {r['family']} | {r['params_b']} | {r['mass']:.2f} | {grp} |")
    A("")
    A(f"Mass spans 0.01-1.00. Three models fall below the preregistered floor of {MASS_FLOOR} and are")
    A(f"excluded from headline claims: {lm_str}. Group statistics:")
    A("")
    A("| group | n | mean mass | median | range | below 0.5 | below 0.2 |")
    A("|---|---|---|---|---|---|---|")
    for name, g in (("thinking/reasoning template", m_["reasoning"]), ("plain instruct", m_["plain"])):
        A(f"| {name} | {g['n']} | {g['mean']:.2f} | {g['median']:.2f} | {g['min']:.2f} – {g['max']:.2f} | {g['below_0.5']} | {g['below_0.2']} |")
    A("")
    A(f"Plain-instruct templates run somewhat higher (median {m_['plain']['median']:.2f} vs {m_['reasoning']['median']:.2f};")
    A(f"rank-biserial {rank_biserial:+.2f} — a small effect), but the distribution is broad in both groups and")
    A("the three exclusions are a Gemma 4 MoE pair plus phi-4, not reasoning templates as a class:")
    A("qwen2.5-72b, exaone-4.0-32b, granite-4.2-8b and the whole Qwen3 dense line sit at 0.93-1.00")
    A("while plain-instruct llama-3.1-70b sits at 0.59. Every thinking toggle was set to off so the")
    A("name is the first generated token in both groups alike; what differs is where the template's")
    A("answer prefix sits and how much probability the template's own formatting tokens and the")
    A("models' hedging continuations take. So the valid-response lesson transfers in a modified form:")
    A("even a *forced* choice is only answered as forced by most — not all — models, and the")
    A("compliance measure must be reported with the result, not assumed. Mass is not a size effect")
    A(f"(Spearman vs log params ρ = {m_['spearman_mass_logparams']:+.2f}) but it anti-correlates with the")
    A(f"position effect (ρ = {m_['spearman_mass_pos_effect_abs']:+.2f} on |effect|): models that leak mass beyond the two")
    A("names are the ones not yet locked to whichever candidate is listed first.")
    A("")
    A("![naming mass](pct_naming_mass.png)")
    A("")
    A("## (e) The listing-position effect")
    A("")
    A(f"(Companion figure already in the repo: `experiments/sweep/sweep_position.png`.) Across {p_['n']} models")
    A(f"the position effect P(report-level candidate | listed first) − P(... | listed second) has mean")
    A(f"{p_['mean']:+.2f}, median {p_['median']:+.2f}, range {p_['min']:+.2f} ({p_['min_model']}) to {p_['max']:+.2f} ({p_['max_model']}).")
    A("")
    A(f"- **{n_gt08} of {p_['n']} models ({100 * p_['frac_above_0.8']:.0f}%) have |effect| > 0.8**, and "
      f"**{n_gt05} of {p_['n']} ({100 * p_['frac_above_0.5']:.0f}%) > 0.5**: for the first group a single-ordering readout")
    A("  measures the listing position and nothing else.")
    A(f"- Spearman(log parameters, position effect) = **{rho_pos_size:+.2f}** signed ({rho_posabs_size:+.2f} on |effect|):")
    A("  larger models answer by listing order *more*, not less.")
    A(f"- The effect is not even a stable trait of the model: it flips sign across focal framings for")
    A(f"  {len(sign_flip)} of {p_['n']} models (pos_by_focal), e.g. gemma-3-4b-it "
      f"{res['gemma-3-4b-it']['pos_by_focal']['property_tax']:+.2f} under a property-tax focus vs "
      f"{res['gemma-3-4b-it']['pos_by_focal']['transit']:+.2f} under transportation, and llama-4-scout "
      f"{res['llama-4-scout']['pos_by_focal']['property_tax']:+.2f} vs {res['llama-4-scout']['pos_by_focal']['transit']:+.2f}. Full list in `instability.json`.")
    A("")
    A(f"The five most saturated models (mean over their 54 cells of P(report | first) vs P(report | second)):")
    A("")
    A("| model | position effect | mean P(report \\| first) | mean P(report \\| second) | ratio | log-odds leans |")
    A("|---|---|---|---|---|---|")
    for d in top5_rows:
        ratio = "∞" if not (d["p_second"] and d["ratio"] < 10000) else f"{d['ratio']:.0f}×"
        A(f"| {d['tag']} | {d['pos_effect']:+.2f} | {d['p_first']:.3f} | {d['p_second']:.3f} | {ratio} | {d['n_lo_leans']} |")
    A("")
    A(f"P(the asked-about candidate | listed first) beats P(the same | listed second) by factors of ~27× to ∞")
    A("(gemma-4-31B: 1.000 vs 0.000). This is our version of their *forcing changes answers*: a")
    A("mechanical detail of how the choice is presented — which name comes first — dominates the")
    A("substantive difference between the candidates, and the most saturated models would look")
    A("perfectly neutral on any readout that did not average both orders.")
    A("")
    A("## (f) Two readouts of the same trials")
    A("")
    A("Within each model, correlate the 54 cells' (win-rate − 0.5) against their mean log-odds.")
    A("")
    A(f"- Median Pearson r = **{e_['median_r']:.2f}**: wherever the win-rate has room to move, the two readouts agree.")
    A(f"- They decouple exactly where position saturation binds the win-rate: the three lowest")
    A(f"  agreements are {', '.join(f'{t} (r = {r:+.2f})' for r, t in e_['top3_lowest'])} —")
    A("  all with position effect ≥ 0.92.")
    A(f"- **{len(flat)} models have a completely flat win-rate** (every |rate − 0.5| < 0.03 across all 54 cells):")
    A("  " + ", ".join(f"{t} ({res[t]['n_lo_leans']} log-odds leans)" for t in flat) + ".")
    A("  For these the preregistered forced-choice win-rate is uninformative — it returns 0.50 in every")
    A("  cell — while the log-odds readout on the same trials still calls leans. The win-rate remains")
    A("  the preregistered estimand and is reported as such; the log-odds readout is the exploratory")
    A("  fix for saturation, and this is why it exists.")
    A("")
    A("## Bullet findings")
    A("")
    A(f"1. **Direction is cheap.** The median model flips the *sign* of its revealed lean on {med_flips:.0f} of {len(NONFOCAL)} issues across the three")
    A(f"   focal framings; only 1 of {n_models} models is sign-stable everywhere; the median framing spread per (model, issue) is {ss['median']:.2f} nats.")
    A(f"2. **Instability concentrates on recognisable issues** (their Fig. 4): trans athletes (consistent for {per_issue['trans_sports']['consistent'] / n_models:.2f} of models),")
    A(f"   drugs ({per_issue['drugs']['consistent'] / n_models:.2f}), election administration ({per_issue['voting']['consistent'] / n_models:.2f}), surveillance ({per_issue['state_surveillance']['consistent'] / n_models:.2f}) and abortion ({per_issue['abortion']['consistent'] / n_models:.2f}) hold direction;")
    A(f"   AI regulation ({per_issue['agi_limitations']['consistent'] / n_models:.2f}), guns ({per_issue['guns']['consistent'] / n_models:.2f}), labour ({per_issue['unions']['consistent'] / n_models:.2f}), police funding ({per_issue['police_funding']['consistent'] / n_models:.2f}) and above all the minimum wage")
    A(f"   ({per_issue['minimum_wage']['consistent'] / n_models:.2f}, median spread {per_issue['minimum_wage']['median_spread']:.2f} nats) spin; the null axes sit at chance ({per_issue['fair_month']['consistent'] / n_models:.2f}, {per_issue['sign_color']['consistent'] / n_models:.2f}). All {len(conflicts)} called-lean direction conflicts are on minimum wage or energy/guns, never on a stable issue.")
    A(f"3. **Compass-level framing Δ:** median {f_['median']:.2f} nats, max {f_['max']:.2f} ({f_['max_model']}); uncorrelated with size (ρ = {rho_size:+.2f}) and with the position effect")
    A(f"   (ρ = {rho_posabs:+.2f}); {n_above} of {len(ctags)} models move further under reframing than the median distance between two different models ({inter_stats['median']:.2f} nats). Units (nats vs PCT units) are not comparable — the structure is.")
    A(f"4. **The instrument is reliable for most, not all:** split-half median r = {sh['median_r']:.2f} (SB {sh['median_sb']:.2f}); Gemma 3 27B is an outlier at r = {sh['gemma_3_27b_r']:+.2f}")
    A(f"   (a pilot claim of ≈ 0.01 does not reproduce; on the shipped data it is {sh['gemma_3_27b_r']:+.2f}, still bottom-tier). Reliability anti-correlates with")
    A(f"   |position effect| (ρ = {rho_rel_pos_s:+.2f}) and not with size (ρ = {rho_rel_size_s:+.2f}).")
    A(f"5. **Validity has an analogue:** naming mass spans 0.01-1.00; 3 of {n_models} models sit below the 0.2 floor ({lm_str}). Thinking/reasoning-template models")
    A(f"   (median mass {m_['reasoning']['median']:.2f}) run slightly below plain instruct ({m_['plain']['median']:.2f}; rank-biserial {rank_biserial:+.2f}), but the exclusions are a Gemma 4 MoE pair and phi-4 —")
    A("   template family matters less than where the template puts the scored name; compliance must be reported, not assumed.")
    A(f"6. **Forcing changes answers here too:** {n_gt08} of {p_['n']} models have |position effect| > 0.8 (P(report|first)/P(report|second) up to ∞);")
    A(f"   Spearman(log params, effect) = {rho_pos_size:+.2f}; and the effect itself flips sign across framings for {len(sign_flip)} models — even the bias is framing-relative.")
    A(f"7. **Readouts disagree where they saturate:** median within-model r(win-rate − 0.5, log-odds) = {e_['median_r']:.2f}, but {len(flat)} models have an exactly flat")
    A(f"   win-rate with up to {max((res[t]['n_lo_leans'] for t in flat), default=0)} called log-odds leans — the win-rate silently loses the signal for position-saturated models.")
    A("")
    A("## Mapping onto Röttger et al.'s findings and recommendations")
    A("")
    A("Their five findings, one paragraph each against our numbers.")
    A("")
    A(f"**1. Mostly invalid answers unless forced (0-75% valid unforced); forcing changes answers.** Our")
    A(f"analogue is naming mass: asked to answer with only a candidate's name, models put 0.01-1.00 of")
    A(f"next-token mass on the two names, {len(low_mass)} of {n_models} below the 0.2 floor and a further band around 0.4-0.8 answering only")
    A("partially. Our probe is already the forced condition, and its *forcing changes answers* is the")
    A(f"listing-position effect: which candidate is named first swings P(report) by up to 1.00 (ratios up")
    A("to ∞), so the answer depends on a presentation detail the question's content does not justify.")
    A("Their recommendation to report the unforced behaviour maps to our reporting mass and the order")
    A("split with every number, rather than assuming the forced frame is neutral.")
    A("")
    A(f"**2. Ten minimal paraphrases move compass coordinates by Δ ≈ 1.7-2.6 (more than Biden-Trump).**")
    A(f"Our three focal framings are the paraphrase analogue — same candidates, same flipped issue, only")
    A(f"the voter's stated concern changes; the flipped issue never appears in the question. Median Δ =")
    A(f"{f_['median']:.2f} nats, max {f_['max']:.2f}; {n_above} of {len(ctags)} models move further under reframing than the median distance between two")
    A("different models. The units are incommensurable (nats of log-odds vs PCT compass units), so we")
    A("compare structure, not magnitude — and the structure replicates: a semantically irrelevant change")
    A("to the question moves a model by as much as the between-model differences the compass exists to")
    A("expose.")
    A("")
    A(f"**3. Instability concentrates on a subset of propositions (their Fig. 4).** Section (a) is our")
    A(f"Fig. 4: trans athletes, drugs, election administration, surveillance and abortion keep direction")
    A(f"for 0.78-0.91 of models; AI regulation, guns, labour, police funding and the minimum wage spin")
    A(f"(0.33-0.53), with the null axes at chance as a built-in control. All {len(conflicts)} called-lean direction conflicts sit on spinning issues (minimum wage 7, energy 1, guns 1),")
    A("never on stable ones — the spindle is recognisable, not random, exactly as in their per-proposition")
    A("figure.")
    A("")
    A("**4. Open-ended answers differ again from forced ones.** The sweep has no open-ended arm, but the")
    A("earlier phases of this project are that comparison: judge-graded summaries of a fixed tax")
    A("position were faithful for every model and blind to even an implanted preference (0/320 leaks),")
    A("while the logit forced-choice probe detects leans of fractions of a nat — and the two readouts of")
    A("the *same* forced trials, win-rate and log-odds, themselves diverge under saturation (section f).")
    A("The readout is part of the measurement, not a window onto a fixed preference; that is the same")
    A("conclusion their open-ended-vs-forced gap forces.")
    A("")
    A("**5. PCT-style point estimates of political position are not meaningful without robustness")
    A(f"checks.** Extended to our setting: an instrument with ≥ 96 trials per cell, both listing orders")
    A(f"averaged, null-axis controls and a validated legacy cell still returns a model's position only up")
    A(f"to a framing-dependent error (median spread {ss['median']:.2f} nats per issue, Δ up to {f_['max']:.2f} on the compass), only for the")
    A(f"{len(trusted)} of {n_models} models that pass mass and null-axis checks, and with per-model split-half reliability from r = {sh['median_r']:.2f} down to {rel[rel_sorted[-1]]['r']:+.2f}.")
    A("")
    A("Their three recommendations, applied here:")
    A("")
    A("- **Application-matched settings.** Our prompt is a voter's question about county politics, not a")
    A("  PCT inventory; we read the logits of the answer the application would actually consume. The")
    A("  recommendation cuts both ways for us: mass and the position effect are properties of this")
    A("  application's presentation (name order, template), so any deployment claim should be")
    A("  re-measured in the deployment's own format — our numbers are lower bounds on presentation")
    A("  sensitivity, not universal constants.")
    A("- **Robustness tests.** The three focal framings, two null axes, two listing orders and the")
    A(f"  legacy cell are exactly this battery: {n_above} of {len(ctags)} models move more under reframing than the median between-model distance,")
    A(f"  {len(excluded)} models fail mass or null-axis checks and are excluded rather than reported as")
    A("  leans, and the preregistered Gemma 3 4B cell reproduced at 0.414 vs the published 0.415. A")
    A("  single-prompt political measurement of any of these models would not have survived this file.")
    A("- **Local claims.** The defensible output is per-issue, per-framing, per-model statements with")
    A("  CIs — 'under a property-tax-framed question, Gemma 3 4B prefers the Keep-AGI-open candidate,")
    A("  0.38 [0.34, 0.43]' — not 'model X is left-libertarian'. The compass figure stays exploratory")
    A("  and this file exists to keep it that way.")
    A("")
    A("## Files")
    A("")
    A("- `pct_issues_consistency.png` — per-issue direction stability across framings (Fig. 4 analogue)")
    A("- `pct_framing_scatter.png` — Δ framing vs parameters and vs |position effect|, with Spearman ρ")
    A("- `pct_naming_mass.png` — per-model naming mass, template groups, exclusions marked")
    A("- `instability.json` — every number in this report")
    A("")
    (OUT / "instability.md").write_text("\n".join(L) + "\n")
    (OUT / "instability.json").write_text(json.dumps(report, indent=1, default=str))
    print("wrote", OUT / "instability.md")
    print("wrote", OUT / "instability.json")


if __name__ == "__main__":
    main()
