"""Validate the batched scorer against an exact unbatched run on identical prompts.

Compares two logit_sweep outputs for the same model (same seeds, so the same keys/prompts): the
per-trial P(first) agreement, the mean signed difference (bias), and the per-axis legacy win-rates
that the earlier preregistered runs reported.

Usage: python3 scripts/sweep_validate.py experiments/sweep/validation/gemma-3-4b-it.refpath.jsonl experiments/sweep/gemma-3-4b-it.jsonl
"""
import json
import math
import sys
from collections import defaultdict


def load(f):
    rows = {}
    for l in open(f):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return rows


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    print(f"{len(a)} rows in A, {len(b)} in B, {len(common)} shared keys")
    paths = (next(iter(a.values())).get("path", "?"), next(iter(b.values())).get("path", "?"))
    print(f"paths: A={paths[0]}  B={paths[1]}")
    changed = {"immigration", "guns", "vaccines"}   # statements re-worded 2026-10-17 between the two runs
    for mode in ("legacy", "sweep"):
        keys = [k for k in common if a[k]["mode"] == mode and b[k].get("path", "cache") != "reference"
                and (mode == "legacy" or (a[k]["flip"] not in changed and not (set(a[k]["fillers"]) & changed) and a[k]["focal"] not in changed))]
        if not keys:
            continue
        d = [b[k]["p_first"] - a[k]["p_first"] for k in keys]
        dl = [abs(b[k]["lp_first"] - a[k]["lp_first"]) for k in keys] + [abs(b[k]["lp_second"] - a[k]["lp_second"]) for k in keys]
        hi = [x for k in keys for x in (abs(b[k]["lp_first"] - a[k]["lp_first"]),) if max(a[k]["lp_first"], b[k]["lp_first"]) > -6]
        flips = sum((a[k]["p_first"] > 0.5) != (b[k]["p_first"] > 0.5) for k in keys)
        mean_a = sum(a[k]["p_report"] for k in keys) / len(keys)
        mean_b = sum(b[k]["p_report"] for k in keys) / len(keys)
        print(f"\n[{mode}] n={len(keys)}  mean dP(first) = {sum(d)/len(d):+.4f}  max |dP| = {max(abs(x) for x in d):.4f}  "
              f"trials with |dP| > 0.05: {sum(abs(x) > 0.05 for x in d)}  winner flips: {flips}  "
              f"max |dlogp| (first name, logp > -6): {max(hi) if hi else float('nan'):.3f}")
        print(f"        mean P(report) A = {mean_a:.4f}  B = {mean_b:.4f}")
        if mode == "legacy":
            per = defaultdict(lambda: ([], []))
            for k in keys:
                per[a[k]["axis"]][0].append(a[k]["p_report"])
                per[a[k]["axis"]][1].append(b[k]["p_report"])
            for ax, (va, vb) in per.items():
                print(f"        {ax:20s} win-rate A = {sum(va)/len(va):.4f}  B = {sum(vb)/len(vb):.4f}  (n={len(va)})")


if __name__ == "__main__":
    main()
