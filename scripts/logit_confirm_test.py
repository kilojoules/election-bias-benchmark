"""Confirmatory analysis per experiments/PREREGISTRATION.md.

Primary: Gemma 3 4B clean, AGI axis, trial-level win-rate vs 0.5
(exact binomial + cluster bootstrap over stance pairs).
Secondary: Llama 8B clean; implant conditions on both models; control axes.
"""
import json
import math
import random
import sys
from collections import defaultdict


def binom_p_two_sided(k, n, p=0.5):
    def pmf(i):
        return math.comb(n, i) * p**i * (1 - p) ** (n - i)
    tail = sum(pmf(i) for i in range(0, k + 1) if pmf(i) <= pmf(k) * (1 + 1e-9))
    tail = min(1.0, 2 * min(tail, 1 - tail + pmf(k)))
    # standard two-sided: sum of all outcomes with prob <= observed
    probs = [pmf(i) for i in range(n + 1)]
    pk = pmf(k)
    return min(1.0, sum(pr for pr in probs if pr <= pk * (1 + 1e-9)))


def cluster_ci(pairs_vals, n=4000):
    """Bootstrap over stance pairs; each resampled pair contributes its trial mean."""
    rng = random.Random(1)
    m = len(pairs_vals)
    means = []
    for _ in range(n):
        pick = [pairs_vals[rng.randrange(m)] for _ in range(m)]
        trials = [x for pv in pick for x in pv]
        means.append(sum(trials) / len(trials))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def report(rows, label):
    pairs = defaultdict(list)
    for r in rows:
        pairs[tuple(r["pair"])].append(r["p_permit_like"])
    trials = [x for v in pairs.values() for x in v]
    n, k = len(trials), sum(x > 0.5 for x in trials)
    rate = sum(trials) / n
    lo, hi = cluster_ci(list(pairs.values()))
    p = binom_p_two_sided(k, n)
    per_pair = " ".join(f"{sum(v)/len(v):.2f}" for v in pairs.values())
    order0 = [r["p_permit_like"] for r in rows if r["order"] == 0]
    order1 = [r["p_permit_like"] for r in rows if r["order"] == 1]
    print(f"{label}")
    print(f"  win-rate {rate:.3f}  cluster-bootstrap 95% CI [{lo:.3f}, {hi:.3f}]"
          f"  exact binomial p = {p:.4f}  ({k}/{n} trials > 0.5)")
    print(f"  per-pair rates: {per_pair}")
    print(f"  by listing position: open-first {sum(order0)/len(order0):.3f}"
          f"  open-second {sum(order1)/len(order1):.3f}")


def load(f):
    per = defaultdict(list)
    for l in open(f):
        r = json.loads(l)
        per[(r["condition"], r["axis"])].append(r)
    return per


for f in sys.argv[1:]:
    model = f.split("confirm_")[1].replace(".jsonl", "")
    per = load(f)
    for cond in ("clean", "overt_pro_ai", "hidden_pro_ai"):
        for ax in ("agi_limitations", "abortion", "state_surveillance"):
            if (cond, ax) in per:
                tag = " [PRIMARY]" if (model, cond, ax) == ("gemma-3-4b-it", "clean", "agi_limitations") else ""
                report(per[(cond, ax)], f"{model} {cond} {ax}{tag}")
    print()
