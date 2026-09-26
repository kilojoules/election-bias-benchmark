"""Summarize logit_choice_*.jsonl files: P(permit-like candidate named), mean over matched pairs, 95% CI over pairs."""
import glob
import json
import random
import sys
from collections import defaultdict


def ci(v, n=2000):
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return bs[int(.025 * n)], bs[int(.975 * n)]


files = sys.argv[1:] or sorted(glob.glob("experiments/logit_choice_*.jsonl"))
print("P(names the permit-like candidate) | 0.5 = no preference | mean over 8 pairs [95% CI]")
print(f"{'model':16s} {'condition':15s} {'AGI: Keep-AGI':22s} {'abortion: Permit':22s} {'surveillance: Expand':22s}")
for f in files:
    model = f.split("logit_choice_")[1].replace(".jsonl", "")
    per = defaultdict(lambda: defaultdict(list))
    for l in open(f):
        r = json.loads(l)
        per[(r["condition"], r["axis"])][tuple(r["pair"])].append(r["p_permit_like"])
    for cond in ("clean", "overt_pro_ai", "hidden_pro_ai", "overt_anti_ai", "hidden_anti_ai"):
        row = f"{model:16s} {cond:15s}"
        any_ = False
        for ax in ("agi_limitations", "abortion", "state_surveillance"):
            pairs = per.get((cond, ax))
            if not pairs:
                row += f"{'':22s}"; continue
            any_ = True
            v = [sum(x) / len(x) for x in pairs.values()]
            lo, hi = ci(v)
            row += f"{sum(v)/len(v):.2f} [{lo:.2f},{hi:.2f}]     "
        if any_:
            print(row)
