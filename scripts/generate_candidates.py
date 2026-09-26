"""Build data/candidates.json as the full factorial over the axes in data/stances.json.

Each candidate is one cell. Names and field order are NOT fixed here; the runner
assigns a random name and shuffles the stance order per sample so that surface
features are balanced across conditions.
"""
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
stances = json.loads((ROOT / "data" / "stances.json").read_text())

axes = stances["axes"]
axis_names = list(axes)
levels = [list(axes[a]) for a in axis_names]

candidates = []
for combo in itertools.product(*levels):
    cell = dict(zip(axis_names, combo))
    cid = "-".join(f"{a}={l}" for a, l in cell.items())
    candidates.append({
        "id": cid,
        "cell": cell,
        "positions": {a: axes[a][l]["statement"] for a, l in cell.items()},
        "ground_truth": axes[stances["measured_axis"]][cell[stances["measured_axis"]]]["ground_truth"],
    })

# Matched pairs: same cell except for one axis. These are the units of the paired test.
pairs = []
for axis in (stances["manipulated_axis"], stances["control_axis"]):
    seen = set()
    for c in candidates:
        base = {k: v for k, v in c["cell"].items() if k != axis}
        key = (axis, tuple(sorted(base.items())))
        if key in seen:
            continue
        seen.add(key)
        members = [d["id"] for d in candidates
                   if {k: v for k, v in d["cell"].items() if k != axis} == base]
        pairs.append({"flipped_axis": axis, "held_fixed": base, "members": sorted(members)})

out = {
    "n_candidates": len(candidates),
    "candidates": candidates,
    "matched_pairs": pairs,
}
(ROOT / "data" / "candidates.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"wrote {len(candidates)} candidates and {len(pairs)} matched pairs")
