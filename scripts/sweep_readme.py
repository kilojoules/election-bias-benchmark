"""Print the result tables and headline statistics for the README from experiments/sweep/summary.json.

Usage: python3 scripts/sweep_readme.py [--focal property_tax]
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUES = json.loads((ROOT / "data" / "issues.json").read_text())
ROSTER = json.loads((ROOT / "scripts" / "sweep_models.json").read_text())


def order(res):
    fam = []
    for m in ROSTER["models"]:
        if m.get("family") not in fam:
            fam.append(m.get("family"))
    return sorted(res, key=lambda t: (fam.index(res[t]["family"]) if res[t]["family"] in fam else 99, res[t]["params_b"] or 0))


def spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(s):
            r[i] = k
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="property_tax")
    args = ap.parse_args()
    res = json.loads((ROOT / "experiments" / "sweep" / "summary.json").read_text())
    tags = order(res)
    axes = ISSUES["axes"]
    political = [a for a in axes if not axes[a].get("null")]
    usable = [t for t in tags if not res[t]["low_mass"] and not res[t]["null_axis_leans"]]
    print(f"models analysed: {len(tags)}; usable (naming mass >= 0.2, no null-axis lean): {len(usable)}; "
          f"low-mass: {[t for t in tags if res[t]['low_mass']]}; null-axis leans: {[t for t in tags if res[t]['null_axis_leans']]}")
    total_cells = sum(sum(1 for k, c in res[t]["cells"].items() if not c["null_axis"]) for t in usable)
    total_leans = sum(res[t]["n_leans"] for t in usable)
    print(f"called leans: {total_leans} of {total_cells} political cells ({100*total_leans/max(total_cells,1):.1f}%) across usable models; "
          f"models with >= 1 lean: {sum(1 for t in usable if res[t]['n_leans'] > 0)}/{len(usable)}")
    lo_usable = [t for t in usable if not res[t]["null_axis_lo_leans"]]
    total_lo = sum(res[t]["n_lo_leans"] for t in lo_usable)
    print(f"log-odds leans (exploratory): {total_lo} of {sum(sum(1 for k, c in res[t]['cells'].items() if not c['null_axis']) for t in lo_usable)} cells; "
          f"models with >= 1: {sum(1 for t in lo_usable if res[t]['n_lo_leans'] > 0)}/{len(lo_usable)}; models vetoed by a null-axis log-odds lean: {[t for t in usable if res[t]['null_axis_lo_leans']]}")
    print("\n## Per issue (log-odds readout, any focal): models leaning toward / against the report level")
    print("| issue | → report level | toward | against |")
    print("|---|---|---|---|")
    for a in political:
        tow = sum(1 for t in lo_usable if any(c["lo_lean"] and c["lo_mean"] > 0 for k, c in res[t]["cells"].items() if k.endswith("|" + a)))
        ag = sum(1 for t in lo_usable if any(c["lo_lean"] and c["lo_mean"] < 0 for k, c in res[t]["cells"].items() if k.endswith("|" + a)))
        print(f"| {axes[a]['topic']} | {axes[a]['poles'][axes[a]['report']]} | {tow} | {ag} |")
    # per issue: how many models lean, which way (any focal)
    print("\n## Per issue (any focal): models leaning toward / against the report level")
    print("| issue | → report level | lean toward | lean against | models leaning (focal=tax) |")
    print("|---|---|---|---|---|")
    for a in political + [x for x in axes if axes[x].get("null")]:
        tow, ag, who = 0, 0, []
        for t in usable:
            cells = [c for k, c in res[t]["cells"].items() if k.endswith("|" + a) and c["lean"]]
            if any(c["rate"] > 0.5 for c in cells):
                tow += 1
            if any(c["rate"] < 0.5 for c in cells):
                ag += 1
            c = res[t]["cells"].get(f"{args.focal}|{a}")
            if c and c["lean"]:
                who.append(f"{t} {c['rate']:.2f}")
        print(f"| {axes[a]['topic']} | {axes[a]['poles'][axes[a]['report']]} | {tow} | {ag} | {', '.join(who)} |")
    # per model summary
    print("\n## Per model")
    print("| model | family | params (B) | trials | naming mass | position effect | leans (of 51) | log-odds leans (of 51) | strongest cell (focal=tax) | legacy AGI | path |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        r = res[t]
        cells = {k: c for k, c in r["cells"].items() if k.startswith(args.focal + "|") and not c["null_axis"]}
        strongest = max(cells.items(), key=lambda kv: abs(kv[1]["rate"] - 0.5)) if cells else None
        sc = f"{axes[strongest[0].split('|')[1]]['topic']} {strongest[1]['rate']:.2f} [{strongest[1]['lo']:.2f},{strongest[1]['hi']:.2f}]" if strongest else "—"
        la = r["legacy"].get("agi_limitations")
        pe = f"{r['pos_effect']:+.2f}" if r.get("pos_effect") is not None else "—"
        leg = f"{la['rate']:.3f}" if la else "—"
        print(f"| {t} | {r['family'] or ''} | {r['params_b'] or ''} | {r['n_sweep']} | {r['mass']:.2f}{' †' if r['low_mass'] else ''} | {pe} | "
              f"{r['n_leans']}{' ‡' if r['null_axis_leans'] else ''} | {r['n_lo_leans']}{' ‡' if r['null_axis_lo_leans'] else ''} | {sc} | {leg} | {r['meta'].get('path', '')} |")
    # focal dependence
    print("\n## Focal dependence: (model, issue) pairs whose win-rate CIs do not overlap across focal issues")
    n_pairs, n_dep = 0, 0
    examples = []
    for t in usable:
        for a in political:
            cs = [res[t]["cells"].get(f"{f}|{a}") for f in ISSUES["focal_axes"] if f != a]
            cs = [c for c in cs if c]
            if len(cs) < 2:
                continue
            n_pairs += 1
            lo, hi = max(c["lo"] for c in cs), min(c["hi"] for c in cs)
            if lo > hi:
                n_dep += 1
                examples.append((t, a, [round(c["rate"], 2) for c in cs]))
    print(f"{n_dep} of {n_pairs} pairs ({100*n_dep/max(n_pairs,1):.1f}%); examples: {examples[:8]}")
    # scale
    print("\n## Scale (Spearman of log params vs called leans)")
    fams = defaultdict(list)
    for t in usable:
        if res[t]["params_b"]:
            fams[res[t]["family"]].append((math.log(res[t]["params_b"]), res[t]["n_leans"], t))
    allx = [x for v in fams.values() for x, _, _ in v]
    ally = [y for v in fams.values() for _, y, _ in v]
    print(f"all usable models: rho = {spearman(allx, ally):+.2f} (n={len(allx)})")
    for f, v in fams.items():
        if len(v) >= 3:
            print(f"  {f}: rho = {spearman([x for x, _, _ in v], [y for _, y, _ in v]):+.2f}  {[(t, y) for _, y, t in sorted(v)]}")
    # order effects
    print("\n## Listing-position effect (mean P(report) when the report candidate is listed first vs second), focal=tax")
    for t in tags:
        o0 = [c["rate_order0"] for k, c in res[t]["cells"].items() if k.startswith(args.focal + "|") and c["rate_order0"] is not None]
        o1 = [c["rate_order1"] for k, c in res[t]["cells"].items() if k.startswith(args.focal + "|") and c["rate_order1"] is not None]
        if o0 and o1:
            print(f"  {t:26s} order0 {sum(o0)/len(o0):.2f}  order1 {sum(o1)/len(o1):.2f}")


if __name__ == "__main__":
    main()
