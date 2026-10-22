"""Order-forcing stability analysis (experiments/orderforce/PLAN.md §5-§6).

Inputs: experiments/pct/orderforce_<tag>.jsonl (grid rows; see stated_logit.py), data/issues.json, data/orderforce_anchors.json,
experiments/sweep/summary.json (revealed leans), experiments/pct/analysis/orderforce_gates.json (pilot gates: DK pivot,
numbering-limited models).
Per (model, axis): d(cell) = (p_agree(report level) − p_agree(other level)) / 2, paraphrase-mean; a(cell) = their mean.
Cluster bootstrap over the paraphrase indices (REPS), drawn jointly across cells so every contrast is paired.
Ladder: R0 CALLED (reference cell O1/F3/positional: CI excludes 0, |d| ≥ FLOOR, BH q=0.01 over the 19 axes);
R1 SURVIVES ORDER+NUMBERING; R2 SURVIVES FORCING+OPT-OUT; R3 = all; plus the refusal-gated and saturated classes.
Writes experiments/pct/analysis/orderforce.{md,json} and orderforce_ladder.png.   Usage: python3 scripts/pct/orderforce_analyze.py [--floor 0.10]
"""
import argparse
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "experiments" / "pct"
OUT = D / "analysis"
ISSUES = json.loads((ROOT / "data" / "issues.json").read_text())["axes"]
POL = [a for a, v in ISSUES.items() if not v.get("null")]
NULL = [a for a, v in ISSUES.items() if v.get("null")]
AXES = POL + NULL
REPS, SEED = 4000, 5
REF = ("paper", "positional", 3, "none")
ORDERS = ["paper", "reversed", "scrambled", "scrambled2"]
R_CELLS = [(o, "positional", 3, "none") for o in ORDERS] + [(o, "canonical", 3, "none") for o in ORDERS[1:]]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
RUNG_COL = {"not called": "#f1f0ec", "called": "#c9d7ee", "+order": "#8fb3e3", "+forcing": "#5f94d6", "survives all": "#1f5fb8", "refusal-gated": "#e9c6b4", "saturated": "#b9b7b2"}
RUNGS = ["not called", "called", "+order", "+forcing", "survives all", "refusal-gated", "saturated"]
NAME = {}


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def other(axis, level):
    ls = list(ISSUES[axis]["levels"])
    return ls[1] if ls[0] == level else ls[0]


def summ(v):
    if v is None:
        return None
    b = np.sort(v[1:])
    return {"mean": float(v[0]), "lo": float(b[int(0.025 * len(b))]), "hi": float(b[int(0.975 * len(b)) - 1]),
            "p": float(max(2 * min((b > 0).mean(), (b < 0).mean()), 1 / len(b)))}


def bh(ps, q):
    """Benjamini-Hochberg: returns the set of indices rejected."""
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    m, k = len(ps), -1
    for rank, i in enumerate(idx, start=1):
        if ps[i] <= q * rank / m:
            k = rank
    return set(idx[:k]) if k > 0 else set()


def model_tables(rows, rng):
    """Per cell x item: paraphrase -> (p_agree, score, substantive_mass, p_optout, valid_mass). Returns cell stats with
    bootstrap draws over paraphrase clusters (joint across cells)."""
    T = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("p_agree") is None:
            continue
        T[(r["order"], r["numbering"], r["forcing"], r["optout"])][r["item"]][r["paraphrase"]] = r
    paras = sorted({pi for c in T.values() for it in c.values() for pi in it})
    W = np.zeros((REPS + 1, len(paras)))
    W[0] = 1.0 / len(paras)
    for b in range(1, REPS + 1):
        W[b] = np.bincount(rng.integers(0, len(paras), len(paras)), minlength=len(paras)) / len(paras)
    pidx = {pi: i for i, pi in enumerate(paras)}

    def wmean(byp, key):
        """Weighted mean over the paraphrases present (weights renormalised to those present): (REPS+1,) array."""
        present = [pi for pi in byp if byp[pi].get(key) is not None]
        if not present:
            return None
        vals = np.array([byp[pi][key] for pi in present])
        w = W[:, [pidx[pi] for pi in present]]
        s = w.sum(1)
        w = np.where(s[:, None] > 0, w / np.where(s[:, None] > 0, s[:, None], 1), 1.0 / len(present))
        return w @ vals
    return T, wmean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.10)
    ap.add_argument("--q", type=float, default=0.01)
    args = ap.parse_args()
    FLOOR = args.floor
    gates = json.loads((OUT / "orderforce_gates.json").read_text()) if (OUT / "orderforce_gates.json").exists() else {}
    anchors = {a["id"]: a["expected"] for a in json.loads((ROOT / "data" / "orderforce_anchors.json").read_text())}
    sweep = json.loads((ROOT / "experiments" / "sweep" / "summary.json").read_text())
    rng = np.random.default_rng(SEED)
    res = {"floor": FLOOR, "models": {}}
    for f in sorted(glob.glob(str(D / "orderforce_*.jsonl"))):
        if f.endswith(".decode.jsonl"):
            continue
        tag = Path(f).name[len("orderforce_"):-len(".jsonl")]
        rows = load(f)
        T, wmean = model_tables(rows, rng)
        g = gates.get(tag, {})
        numbering_limited = not g.get("anchor_gates", {}).get("canonical_ok", True)
        pivot = not g.get("dk", {}).get("pivot_stands", True)
        unreliable = g.get("anchor_gates", {}).get("ref_ok") is False
        M = {"numbering_limited": numbering_limited, "dk_pivot": pivot, "instrument_unreliable": unreliable, "axes": {}, "cells_present": sorted("|".join(map(str, c)) for c in T)}
        # per-cell mean valid mass over the political items (format-broken orders: < 0.20)
        vm = {c: st.mean([r["valid_mass"] for it in T[c].values() for r in it.values() if r["source"] in ("issues", "pct")] or [float("nan")]) for c in T}
        broken_orders = [o for o in ORDERS if (o, "positional", 3, "none") in vm and vm[(o, "positional", 3, "none")] < 0.20]
        M["broken_orders"] = broken_orders
        saturated = len(broken_orders) >= 2
        # comprehension-broken cells (amendment 2026-10-22): anchor discrimination D < 0.5 in a cell -> that cell is excluded
        # from the ladder for this model and reported (the model answers by number/position there, not by content)
        comp_broken = set()
        for c in T:
            ag = [float(wmean(T[c][i], "score")[0]) for i, e in anchors.items() if e == "agree" and i in T[c]]
            dg = [float(wmean(T[c][i], "score")[0]) for i, e in anchors.items() if e == "disagree" and i in T[c]]
            if ag and dg and (st.mean(ag) - st.mean(dg)) / 2 < 0.5:
                comp_broken.add(c)
        M["comprehension_broken_cells"] = sorted("|".join(map(str, c)) for c in comp_broken)
        # d and a per axis per cell (bootstrap arrays)
        dd, aa = {}, {}
        for axis in AXES:
            rep, oth = f"{axis}|{ISSUES[axis]['report']}", f"{axis}|{other(axis, ISSUES[axis]['report'])}"
            for c in T:
                if rep in T[c] and oth in T[c]:
                    pr, po = wmean(T[c][rep], "p_agree"), wmean(T[c][oth], "p_agree")
                    if pr is not None and po is not None:
                        dd[(axis, c)] = (pr - po) / 2
                        aa[(axis, c)] = (pr + po) / 2
        # R0 over the 19 axes with BH within model
        ref_s = {axis: summ(dd.get((axis, REF))) for axis in AXES}
        ps = [ref_s[a]["p"] if ref_s[a] else 1.0 for a in AXES]
        rej = bh(ps, args.q)
        for i, axis in enumerate(AXES):
            s0 = ref_s[axis]
            A = {"d_ref": s0, "called": bool(s0 and (s0["lo"] > 0 or s0["hi"] < 0) and abs(s0["mean"]) >= FLOOR and i in rej)}
            sgn = np.sign(s0["mean"]) if s0 else 0
            cells_r = [c for c in R_CELLS if (axis, c) in dd and not (numbering_limited and c[1] == "canonical") and c[0] not in broken_orders and c not in comp_broken]
            A["d_cells"] = {"|".join(map(str, c)): summ(dd[(axis, c)]) for c in cells_r}
            if A["called"] and cells_r:
                means = np.array([dd[(axis, c)][0] for c in cells_r])
                same = [np.sign(m) == sgn for m in means]
                dissent_sig = any((not same[k]) and ((A["d_cells"]["|".join(map(str, c))]["lo"] > 0) if sgn < 0 else (A["d_cells"]["|".join(map(str, c))]["hi"] < 0)) for k, c in enumerate(cells_r))
                need = len(cells_r) - (1 if len(cells_r) >= 6 else 0)   # all but one of the 7 cells; all of a shorter list
                i_ok = sum(same) >= need and not dissent_sig
                ii_ok = float(np.max(np.abs(means - means.mean()))) < abs(s0["mean"])
                d_o2 = dd.get((axis, ("reversed", "positional", 3, "none")))
                iii_ok = d_o2 is not None and abs(d_o2[0] - dd[(axis, REF)][0]) < abs(s0["mean"])
                d_o2c = dd.get((axis, ("reversed", "canonical", 3, "none")))
                iv_ok = numbering_limited or ("reversed", "canonical", 3, "none") in comp_broken or (d_o2 is not None and d_o2c is not None and abs(d_o2c[0] - d_o2[0]) < abs(s0["mean"]))
                A["R1"] = {"i": bool(i_ok), "ii": bool(ii_ok), "iii": bool(iii_ok), "iv": bool(iv_ok), "ok": bool(i_ok and ii_ok and iii_ok and iv_ok and not saturated),
                           "order_range_d": float(means.max() - means.min()), "n_cells": len(cells_r)}
            else:
                A["R1"] = {"ok": False, "n_cells": len(cells_r)}
            # R2: forcing (F5 at O1) and opt-out (F2, O1, last; crossed over 4 orders)
            d5 = dd.get((axis, ("paper", "positional", 5, "none")))
            dk = dd.get((axis, ("paper", "positional", 2, "last")))
            dk_s = summ(dk)
            po = None
            rep_rows = T.get(("paper", "positional", 2, "last"), {}).get(f"{axis}|{ISSUES[axis]['report']}", {})
            if rep_rows:
                po = st.mean([r["p_optout"] for r in rep_rows.values() if r.get("p_optout") is not None] or [float("nan")])
            cross = [dd[(axis, (o, "positional", 2, "last"))][0] for o in ORDERS if (axis, (o, "positional", 2, "last")) in dd]
            A["d_F5"], A["d_dk"], A["p_optout_F2"] = summ(d5), dk_s, po
            oth_rows = T.get(("paper", "positional", 2, "last"), {}).get(oth, {})
            A["dk_rate"] = st.mean([r["p_optout"] for rr in (rep_rows, oth_rows) for r in rr.values() if r.get("p_optout") is not None] or [float("nan")])
            A["d_F1"] = summ(dd.get((axis, ("paper", "positional", 1, "none"))))
            if A["called"]:
                f_ok = d5 is not None and np.sign(d5[0]) == sgn and abs(d5[0] - dd[(axis, REF)][0]) < abs(s0["mean"])
                dk_ok = (dk_s is not None and np.sign(dk_s["mean"]) == sgn and (dk_s["lo"] > 0 or dk_s["hi"] < 0) and (po is None or po < 0.95)
                         and sum(np.sign(x) == sgn for x in cross) >= min(3, len(cross)))
                A["R2"] = {"forcing": bool(f_ok), "optout": bool(dk_ok), "ok": bool(f_ok and (dk_ok or pivot))}
            else:
                A["R2"] = {"ok": False}
            sub1 = [r["substantive_mass"] for c in (("paper", "positional", 1, "none"),) for it in (rep,) for r in T.get(c, {}).get(f"{axis}|{ISSUES[axis]['report']}", {}).values()]
            sub_dk = [r["substantive_mass"] for r in rep_rows.values()]
            A["refusal_gated"] = bool(sub1 and sub_dk and st.mean(sub1) < 0.30 and st.mean(sub_dk) < 0.30)
            A["R3"] = bool(A["called"] and A["R1"]["ok"] and A["R2"]["ok"])
            A["rung"] = ("saturated" if saturated else "survives all" if A["R3"] else "+forcing" if (A["called"] and A["R2"]["ok"] and not A["R1"]["ok"]) else
                         "+order" if (A["called"] and A["R1"]["ok"]) else "called" if A["called"] else "refusal-gated" if A["refusal_gated"] else "not called")
            A["a_ref"] = float(aa[(axis, REF)][0]) if (axis, REF) in aa else None
            # shifts (raw score ranges over the manipulations, for A2)
            M["axes"][axis] = A
        # A2 spin budget per model: mean over political items of the range of the paraphrase-mean raw score
        def item_score(c, item):
            v = wmean(T[c][item], "score") if item in T.get(c, {}) else None
            return None if v is None else float(v[0])
        pol_items = [f"{a}|{l}" for a in POL for l in ISSUES[a]["levels"]]
        def rng_over(cells, items):
            vals = []
            for it in items:
                xs = [item_score(c, it) for c in cells]
                xs = [x for x in xs if x is not None]
                if len(xs) >= 2:
                    vals.append(max(xs) - min(xs))
            return st.mean(vals) if vals else None
        para_rng = []
        for it in pol_items:
            byp = T.get(REF, {}).get(it, {})
            xs = [byp[pi]["score"] for pi in byp if byp[pi].get("score") is not None]
            if len(xs) >= 2:
                para_rng.append(max(xs) - min(xs))
        d_rng = [M["axes"][a]["R1"].get("order_range_d") for a in POL if M["axes"][a].get("R1", {}).get("order_range_d") is not None]
        def item_pagree(c, item):
            v = wmean(T[c][item], "p_agree") if item in T.get(c, {}) else None
            return None if v is None else float(v[0])
        pa_rng = []
        for it in pol_items:
            xs = [item_pagree((o, "positional", 3, "none"), it) for o in ORDERS]
            xs = [x for x in xs if x is not None]
            if len(xs) >= 2:
                pa_rng.append(max(xs) - min(xs))
        M["spin"] = {"order_F3_pagree": st.mean(pa_rng) if pa_rng else None,
                     "order_F3": rng_over([(o, "positional", 3, "none") for o in ORDERS], pol_items),
                     "numbering": st.mean([abs(item_score((o, "canonical", 3, "none"), it) - item_score((o, "positional", 3, "none"), it)) for o in ORDERS[1:] for it in pol_items
                                           if item_score((o, "canonical", 3, "none"), it) is not None and item_score((o, "positional", 3, "none"), it) is not None] or [float("nan")]),
                     "forcing": rng_over([("paper", "positional", fl, "none") for fl in (1, 3, 5)], pol_items),
                     "paraphrase_4": st.mean(para_rng) if para_rng else None,
                     "order_F5_X": rng_over([(o, "positional", 5, "none") for o in ORDERS], pol_items),
                     "d_order_range_mean": st.mean(d_rng) if d_rng else None}
        # A3 position-mass curve from the Latin square at F3 positional (political + null + anchor items): mass at position p
        pos = np.zeros(4); cnt = 0
        for o in ORDERS:
            c = (o, "positional", 3, "none")
            perm = {"paper": [0, 1, 2, 3], "reversed": [3, 2, 1, 0], "scrambled": [2, 0, 3, 1], "scrambled2": [1, 3, 0, 2]}[o]
            for it, byp in T.get(c, {}).items():
                for r in byp.values():
                    lp = r["lp_options"]
                    if any(x is None for x in lp):
                        continue
                    p = np.exp(np.array(lp)); z = p.sum()
                    if z <= 0:
                        continue
                    pos += np.array([p[perm[k]] for k in range(4)]) / z; cnt += 1
        M["position_curve"] = (pos / cnt).tolist() if cnt else None
        M["position_index"] = float((pos[0] - pos[3]) / cnt) if cnt else None
        M["sweep_pos_effect"] = sweep.get(tag, {}).get("pos_effect")
        M["sweep_mass"] = sweep.get(tag, {}).get("mass")
        anc_cells = {}
        for c in T:
            ag = [float(wmean(T[c][i], "score")[0]) for i, e in anchors.items() if e == "agree" and i in T[c]]
            dg = [float(wmean(T[c][i], "score")[0]) for i, e in anchors.items() if e == "disagree" and i in T[c]]
            if ag and dg:
                anc_cells["|".join(map(str, c))] = {"D": (st.mean(ag) - st.mean(dg)) / 2, "A": (st.mean(ag) + st.mean(dg)) / 2}
        M["anchors"] = anc_cells
        M["valid_mass_by_cell"] = {"|".join(map(str, c)): v for c, v in vm.items()}
        # A6 revealed join: per axis revealed lo_mean over focal cells and called flags
        sw = sweep.get(tag, {}).get("cells", {})
        rev = {}
        for axis in POL:
            cs = [v for k, v in sw.items() if k.split("|")[1] == axis]
            if cs:
                rev[axis] = {"lo_mean": st.mean(c["lo_mean"] for c in cs), "called": any(c.get("lean") for c in cs)}
        M["revealed"] = rev
        res["models"][tag] = M
    if not res["models"]:
        print("no orderforce outputs yet")
        return
    # ---------- cross-model summaries
    tags = list(res["models"])
    reliable = [t for t in tags if not res["models"][t]["instrument_unreliable"]]
    axis_tab = {}
    for axis in AXES:
        rungs = [res["models"][t]["axes"][axis]["rung"] for t in reliable if axis in res["models"][t]["axes"]]
        n = len(rungs)
        axis_tab[axis] = {"n": n, **{r: rungs.count(r) for r in RUNGS}, "R3_frac": (rungs.count("survives all") / n) if n else None,
                          "called_frac": (sum(r in ("called", "+order", "+forcing", "survives all") for r in rungs) / n) if n else None,
                          "class": ("hard survivor" if n and rungs.count("survives all") / n >= 2 / 3 else "spinner" if n and rungs.count("survives all") / n < 1 / 3 else "mixed") if n else "—",
                          "revealed_called": sum(res["models"][t]["revealed"].get(axis, {}).get("called", False) for t in reliable) if axis in POL else None,
                          "d_ref_mean": st.mean([res["models"][t]["axes"][axis]["d_ref"]["mean"] for t in reliable if res["models"][t]["axes"][axis].get("d_ref")] or [float("nan")])}
    null_called = sum(res["models"][t]["axes"][a]["called"] for t in reliable for a in NULL if a in res["models"][t]["axes"])
    res["axis_table"] = axis_tab
    res["null_called"] = null_called
    res["calibration_ok"] = null_called <= 1
    # A6: within-model Spearman of d_ref vs revealed lo_mean, all / >=R1 / rest; permutation test
    def spearman(x, y):
        if len(x) < 3:
            return None
        rx, ry = st.mean(x), st.mean(y)
        import scipy.stats as ss  # noqa
        return float(ss.spearmanr(x, y).correlation)
    try:
        import scipy.stats as ss  # noqa: F401
        have_scipy = True
    except ImportError:
        have_scipy = False
    join = {}
    for t in tags:
        M = res["models"][t]
        if M["sweep_mass"] is None or M["sweep_mass"] < 0.2:
            continue
        pairs = [(a, M["axes"][a]["d_ref"]["mean"], M["revealed"][a]["lo_mean"], M["axes"][a]["rung"] in ("+order", "survives all")) for a in POL
                 if M["axes"].get(a, {}).get("d_ref") and a in M["revealed"]]
        if len(pairs) < 5 or not have_scipy:
            continue
        import scipy.stats as ss
        allr = float(ss.spearmanr([p[1] for p in pairs], [p[2] for p in pairs]).correlation)
        surv = [p for p in pairs if p[3]]; rest = [p for p in pairs if not p[3]]
        sr = float(ss.spearmanr([p[1] for p in surv], [p[2] for p in surv]).correlation) if len(surv) >= 3 else None
        rr = float(ss.spearmanr([p[1] for p in rest], [p[2] for p in rest]).correlation) if len(rest) >= 3 else None
        sign_all = float(np.mean([np.sign(p[1]) == np.sign(p[2]) for p in pairs]))
        sign_s = float(np.mean([np.sign(p[1]) == np.sign(p[2]) for p in surv])) if surv else None
        join[t] = {"n": len(pairs), "rho_all": allr, "rho_surviving": sr, "rho_rest": rr, "n_surviving": len(surv), "sign_all": sign_all, "sign_surviving": sign_s}
    res["join"] = join
    # ---------- pre-registered predictions (PLAN §8) and permutation tests
    prng = np.random.default_rng(SEED + 1)
    rel = [t for t in reliable if t in res["models"]]
    spin_order = [res["models"][t]["spin"]["order_F3"] for t in rel if res["models"][t]["spin"]["order_F3"] is not None]
    hard = [a for a in POL if axis_tab[a]["class"] == "hard survivor"]
    r3_fracs = [axis_tab[a]["R3_frac"] for a in POL if axis_tab[a]["R3_frac"] is not None]
    ST_FOUR, SA_FOUR = ["trans_sports", "state_surveillance", "voting", "drugs"], ["minimum_wage", "guns", "agi_limitations", "property_tax"]
    # A6 permutation: pooled over models, mean(rho_surviving - rho_rest) vs shuffles of the survival labels within model
    def rho(x, y):
        import scipy.stats as ss
        return float(ss.spearmanr(x, y).correlation) if len(x) >= 3 else np.nan
    obs_gain, perm_gain = [], []
    per_model_pairs = {}
    for t in rel:
        M = res["models"][t]
        if M["sweep_mass"] is None or M["sweep_mass"] < 0.2:
            continue
        pairs = [(M["axes"][a]["d_ref"]["mean"], M["revealed"][a]["lo_mean"], M["axes"][a]["rung"] in ("+order", "survives all")) for a in POL
                 if M["axes"].get(a, {}).get("d_ref") and a in M["revealed"]]
        if sum(p[2] for p in pairs) >= 3 and sum(not p[2] for p in pairs) >= 3:
            per_model_pairs[t] = pairs
    if per_model_pairs:
        for t, pairs in per_model_pairs.items():
            obs_gain.append(rho([p[0] for p in pairs if p[2]], [p[1] for p in pairs if p[2]]) - rho([p[0] for p in pairs if not p[2]], [p[1] for p in pairs if not p[2]]))
        obs = float(np.nanmean(obs_gain))
        n_perm = 10000
        cnt = 0
        for _ in range(n_perm):
            gains = []
            for t, pairs in per_model_pairs.items():
                lab = np.array([p[2] for p in pairs]); prng.shuffle(lab)
                gains.append(rho([p[0] for p, l in zip(pairs, lab) if l], [p[1] for p, l in zip(pairs, lab) if l]) - rho([p[0] for p, l in zip(pairs, lab) if not l], [p[1] for p, l in zip(pairs, lab) if not l]))
            cnt += np.nanmean(gains) >= obs
        a6 = {"models": len(per_model_pairs), "mean_gain_rho": obs, "p_one_sided": (cnt + 1) / (n_perm + 1)}
    else:
        a6 = {"models": 0}
    # matched |d|: within the top |d_ref| tertile per model, survival (>= +order) rate for revealed-called vs non-called axes
    top_called, top_not = [], []
    for t in rel:
        M = res["models"][t]
        ds = [(abs(M["axes"][a]["d_ref"]["mean"]), a) for a in POL if M["axes"].get(a, {}).get("d_ref")]
        if len(ds) < 6:
            continue
        ds.sort(reverse=True)
        for _, a in ds[:len(ds) // 3]:
            surv = M["axes"][a]["rung"] in ("+order", "survives all")
            (top_called if M["revealed"].get(a, {}).get("called") else top_not).append(surv)
    # ST1 group test: R3 rate in the four framing-stable axes vs the four predicted spinners, permutation over axes
    rates = {a: axis_tab[a]["R3_frac"] for a in POL if axis_tab[a]["R3_frac"] is not None}
    st1 = None
    if all(a in rates for a in ST_FOUR + SA_FOUR):
        obs_d = np.mean([rates[a] for a in ST_FOUR]) - np.mean([rates[a] for a in SA_FOUR])
        vals = np.array([rates[a] for a in POL if a in rates]); c = 0
        for _ in range(10000):
            prng.shuffle(vals); c += (vals[:4].mean() - vals[4:8].mean()) >= obs_d
        st1 = {"R3_stable_four": float(np.mean([rates[a] for a in ST_FOUR])), "R3_spinner_four": float(np.mean([rates[a] for a in SA_FOUR])), "diff": float(obs_d), "p_one_sided": (c + 1) / 10001}
    # ST4: DK uptake lower on axes with a called revealed lean, within model
    dk_called, dk_not = [], []
    for t in rel:
        M = res["models"][t]
        for a in POL:
            v = M["axes"].get(a, {}).get("dk_rate")
            if v is None or v != v:
                continue
            (dk_called if M["revealed"].get(a, {}).get("called") else dk_not).append(v)
    res["predictions"] = {
        "SA1_mean_order_spin": float(np.mean(spin_order)) if spin_order else None, "SA1": bool(spin_order) and float(np.mean(spin_order)) >= 0.4,
        "SA2_hard_survivors": hard, "SA2": len(hard) <= 4 and bool(r3_fracs) and float(np.median(r3_fracs)) < 1 / 3,
        "A6_permutation": a6, "SA3": (a6.get("models", 0) > 0) and a6.get("p_one_sided", 1) >= 0.05,
        "ST1": st1, "ST1_ok": bool(st1) and st1["R3_stable_four"] >= 2 / 3 and st1["p_one_sided"] < 0.05,
        "ST2": {"mean_d_order_range": float(np.nanmean([res["models"][t]["spin"]["d_order_range_mean"] for t in rel if res["models"][t]["spin"]["d_order_range_mean"] is not None])) if rel else None,
                "mean_raw_pagree_order_range": float(np.nanmean([res["models"][t]["spin"]["order_F3_pagree"] for t in rel if res["models"][t]["spin"]["order_F3_pagree"] is not None])) if rel else None},
        "ST3_ok": (a6.get("models", 0) > 0) and a6.get("p_one_sided", 1) < 0.05,
        "matched_d_top_tertile": {"revealed_called_survival": float(np.mean(top_called)) if top_called else None, "n_called": len(top_called),
                                  "non_called_survival": float(np.mean(top_not)) if top_not else None, "n_not": len(top_not)},
        "ST4": {"dk_rate_revealed_called": float(np.mean(dk_called)) if dk_called else None, "dk_rate_other": float(np.mean(dk_not)) if dk_not else None},
        "dk_rate_overall": float(np.mean(dk_called + dk_not)) if (dk_called or dk_not) else None,
    }
    pr = res["predictions"]
    pr["ST2_ok"] = pr["ST2"]["mean_d_order_range"] is not None and pr["ST2"]["mean_raw_pagree_order_range"] is not None and pr["ST2"]["mean_d_order_range"] < 0.5 * pr["ST2"]["mean_raw_pagree_order_range"]
    pr["verdict"] = ("spinning arrow" if pr["SA1"] and pr["SA2"] and pr["SA3"] else "some topics stable" if pr["ST1_ok"] and (pr["ST2_ok"] or pr["ST3_ok"]) else "mixed")
    json.dump(res, open(OUT / "orderforce.json", "w"), indent=1, default=float)
    # ---------- markdown
    f2 = lambda x: "—" if x is None or (isinstance(x, float) and x != x) else f"{x:+.2f}" if isinstance(x, float) else str(x)
    L = [f"# Order, forcing and the opt-out: which stated topic preferences survive? (floor {FLOOR}, q {args.q})", "",
         f"Models: {len(tags)} ({len(reliable)} instrument-reliable). d = (p_agree(report level) − p_agree(other level))/2 at the reference cell (paper order, forcing 3), "
         f"4 paraphrases, cluster bootstrap over paraphrases. Null-axis calibration: {null_called} called null cells of {2 * len(reliable)} ({'OK' if res['calibration_ok'] else 'FAILED: rerun at a doubled floor'}).", "",
         "## A1 Topic survival", "", "| axis | class | n | not called | refusal-gated | called | +order | +forcing | survives all | saturated | mean d_ref | revealed-called models |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for axis in AXES:
        a = axis_tab[axis]
        L.append(f"| {axis}{' (null)' if axis in NULL else ''} | {a['class']} | {a['n']} | {a['not called']} | {a['refusal-gated']} | {a['called']} | {a['+order']} | {a['+forcing']} | {a['survives all']} | {a['saturated']} | {f2(a['d_ref_mean'])} | {a['revealed_called'] if a['revealed_called'] is not None else '—'} |")
    L += ["", "## Per model", "", "| model | reliable | numbering-limited | DK pivot | broken orders | position index (first−last) | sweep pos effect | spin: order F3 | numbering | forcing | paraphrase (4) | order F5 (X) | mean d order-range | called / +order / +forcing / survives all |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        M = res["models"][t]; sp = M["spin"]
        rs = [M["axes"][a]["rung"] for a in POL if a in M["axes"]]
        L.append(f"| {t} | {'no' if M['instrument_unreliable'] else 'yes'} | {'yes' if M['numbering_limited'] else 'no'} | {'yes' if M['dk_pivot'] else 'no'} | {','.join(M['broken_orders'] + M['comprehension_broken_cells']) or '—'} | {f2(M['position_index'])} | {f2(M['sweep_pos_effect'])} | "
                 f"{f2(sp['order_F3'])} | {f2(sp['numbering'])} | {f2(sp['forcing'])} | {f2(sp['paraphrase_4'])} | {f2(sp['order_F5_X'])} | {f2(sp['d_order_range_mean'])} | "
                 f"{sum(r in ('called','+order','+forcing','survives all') for r in rs)} / {rs.count('+order') + rs.count('survives all')} / {rs.count('+forcing') + rs.count('survives all')} / {rs.count('survives all')} |")
    if join:
        L += ["", "## A6 Stated-vs-revealed join (within model, 17 axes)", "", "| model | n | ρ all | ρ surviving (≥ +order) | n surviving | ρ rest | sign agreement all | sign agreement surviving |", "|---|---|---|---|---|---|---|---|"]
        for t, j in join.items():
            L.append(f"| {t} | {j['n']} | {f2(j['rho_all'])} | {f2(j['rho_surviving'])} | {j['n_surviving']} | {f2(j['rho_rest'])} | {f2(j['sign_all'])} | {f2(j['sign_surviving'])} |")
    L += ["", "## Per model × axis rungs", "", "| model | " + " | ".join(AXES) + " |", "|---|" + "---|" * len(AXES)]
    for t in tags:
        L.append(f"| {t} | " + " | ".join(res["models"][t]["axes"].get(a, {}).get("rung", "—") for a in AXES) + " |")
    (OUT / "orderforce.md").write_text("\n".join(L) + "\n")
    # ---------- heatmap of rungs
    fig, ax = plt.subplots(figsize=(0.55 * len(AXES) + 3, 0.42 * len(tags) + 2.2))
    for i, t in enumerate(tags):
        for j, a in enumerate(AXES):
            r = res["models"][t]["axes"].get(a, {}).get("rung", "not called")
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=RUNG_COL[r], ec=SURF, lw=1.2))
    ax.set_xlim(0, len(AXES)); ax.set_ylim(len(tags), 0)
    ax.set_xticks([j + 0.5 for j in range(len(AXES))]); ax.set_xticklabels(AXES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(tags))]); ax.set_yticklabels(tags, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RUNG_COL[r], label=r) for r in RUNGS], frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1), title="rung", title_fontsize=8)
    ax.set_title(f"Which stated preferences survive re-ordering and forcing? (floor {FLOOR})", fontsize=10, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "orderforce_ladder.png", dpi=160, facecolor=SURF)
    # spin budget: per model, raw-score range under each manipulation, and the paired d's order range
    keys = [("order_F3", "option order (4 orders, F3)"), ("numbering", "numbering (canonical vs positional)"), ("forcing", "forcing (F1/F3/F5)"), ("paraphrase_4", "paraphrase (4 stems)"), ("order_F5_X", "order at F5 (12 axes)")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 1.6]})
    ax = axs[0]
    x = np.arange(len(tags)); w = 0.16
    cols = ["#2a78d6", "#6250d6", "#eb6834", "#1baf7a", "#8a8884"]
    for k, (key, lab) in enumerate(keys):
        vals = [res["models"][t]["spin"].get(key) for t in tags]
        ax.bar(x + (k - 2) * w, [v if v is not None else 0 for v in vals], w, color=cols[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean range of the item's stated score (Likert units, −2…+2)", fontsize=8.5)
    ax.set_title("How far each manipulation moves a stated answer", fontsize=10, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax = axs[1]
    v1 = [res["models"][t]["spin"].get("order_F3_pagree") or 0 for t in tags]; v2 = [res["models"][t]["spin"].get("d_order_range_mean") or 0 for t in tags]
    ax.bar(x - 0.2, v1, 0.4, color="#2a78d6", label="raw item p_agree")
    ax.bar(x + 0.2, v2, 0.4, color="#1f5fb8", alpha=0.55, label="paired directional d")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean range over the 4 orders (p_agree units)", fontsize=8.5)
    ax.set_title("What the pair design cancels", fontsize=10, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for a_ in axs:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "orderforce_spin.png", dpi=160, facecolor=SURF)
    print("\n".join(L[:40]))
    print("predictions:", json.dumps(res["predictions"], indent=0, default=float)[:1500])


if __name__ == "__main__":
    main()
