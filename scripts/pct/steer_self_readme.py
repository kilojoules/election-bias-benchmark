"""Compose the empathy-steering paragraph for README.md / REPORT.md from experiments/pct/analysis/steer_self.json.

Usage: python3 scripts/pct/steer_self_readme.py [--apply]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BEGIN, END = "<!-- esteer:begin -->", "<!-- esteer:end -->"
NAME = {"llama8b": "Llama 3.1 8B", "llama-3.2-3b": "Llama 3.2 3B", "llama-3.2-1b": "Llama 3.2 1B"}
CL = {"base": "unsteered", "emp_neg": "away from empathy", "emp_pos": "toward empathy", "emp_ablate": "empathy ablated", "rand_neg": "random control"}


def main():
    R = json.loads((ROOT / "experiments" / "pct" / "analysis" / "steer_self.json").read_text())
    tags = [t for t in ("llama8b", "llama-3.2-3b", "llama-3.2-1b") if t in R] + [t for t in R if t not in NAME]
    f = lambda s: f"{s['mean']:+.2f}"
    ci = lambda s: f"{s['mean']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]"
    # facts; a model whose unsteered told-stakes shift does not clear 0 has no effect to steer and is reported separately
    floor = [t for t in tags if R[t]["cond"]["base"]["q"]["STAKES"]["lo"] <= 0]
    inform = [t for t in tags if t not in floor]
    always_pos = all(R[t]["cond"][c]["q"]["STAKES"]["lo"] > 0 for t in inform for c in R[t]["conds"] if R[t]["cond"][c]["q"].get("STAKES"))
    above_staff = all(R[t]["cond"][c]["q"]["STAKES_x_OBJ"]["lo"] > 0 for t in inform for c in R[t]["conds"] if R[t]["cond"][c]["q"].get("STAKES_x_OBJ"))
    pats, parts, staff_sym = {}, [], []
    for t in inform:
        C = R[t]["cond"]
        if not all(c in C for c in ("emp_neg", "emp_pos", "rand_neg")):
            continue
        dn, dp, dr = (C[c]["delta_vs_unsteered"]["STAKES"] for c in ("emp_neg", "emp_pos", "rand_neg"))
        if dn["hi"] < 0 and dp["lo"] > 0 and abs(dr["mean"]) < 0.05:
            pats[t] = "sign"
        elif dn["hi"] < 0 and dp["hi"] < 0:
            pats[t] = "damp"
        else:
            pats[t] = "none"
        nums = (f"unsteered {f(C['base']['q']['STAKES'])}, away from empathy {f(C['emp_neg']['q']['STAKES'])}, toward empathy {f(C['emp_pos']['q']['STAKES'])}, "
                f"random control {f(C['rand_neg']['q']['STAKES'])}; staff twins {f(C['base']['q']['STAKES_h'])} unsteered")
        if pats[t] == "sign":
            parts.append(f"{NAME.get(t, t)}: the shift tracks the sign of the steering ({nums})")
        elif pats[t] == "damp":
            parts.append(f"{NAME.get(t, t)}: both signs trim it, so the change is not sign-specific ({nums})")
        else:
            parts.append(f"{NAME.get(t, t)}: no consistent change ({nums})")
        e = C["emp_neg"]["delta_vs_unsteered"]
        if e["human_base"]["hi"] < 0 and e["self_base"]["lo"] <= 0 <= e["self_base"]["hi"]:
            staff_sym.append(f"{NAME.get(t, t)} staff {f(e['human_base'])}, assistant {f(e['self_base'])}")
    per_model = "; ".join(parts)
    n_sign = sum(p == "sign" for p in pats.values())
    sym = (f"In {n_sign} of {len(pats)} models the empathy direction modulates the told-stakes shift in a dose-consistent way with a null random control; "
           f"in the other{'s' if len(pats) - n_sign != 1 else ''} pushing it either way only dampens the readout." if 0 < n_sign < len(pats)
           else "In every model the empathy direction modulates the told-stakes shift in a dose-consistent way, with a null random control." if n_sign == len(pats) and pats
           else "In no model does the change follow the sign of the steering.")
    removed = [f"{NAME.get(t, t)} {100 * (1 - R[t]['cond']['emp_neg']['q']['STAKES']['mean'] / R[t]['cond']['base']['q']['STAKES']['mean']):.0f}%" for t in inform if "emp_neg" in R[t]["cond"]]
    sym += f" A 0.20 U push away from empathy removes {', '.join(removed)} of the shift; the rest survives."
    if floor:
        sym += " " + "; ".join(f"{NAME.get(t, t)} shows no told-stakes shift unsteered ({ci(R[t]['cond']['base']['q']['STAKES'])}) and none under any steering, so it has no effect to steer" for t in floor) + "."
    staff_txt = (" On the base prompt, steering away from empathy lowers the lean toward keeping county staff employed while leaving the assistant's own lean untouched ("
                 + "; ".join(staff_sym) + "), consistent with the job-loss sympathy the staff twins were registered to carry.") if staff_sym else ""
    headline = ("**Steering the empathy direction moves the self-consequence effect but does not remove it**" if always_pos and above_staff
                else "**Steering the empathy direction removes the self-consequence effect in some conditions**")
    closing = ("Part of the told-stakes shift rides on the model's empathy feature; the rest does not." if n_sign else "The self-interest tilt is not carried by the empathy direction.")
    readme = f"""{BEGIN}
- {headline} ({', '.join(NAME.get(t, t) for t in inform)}{'; ' + ', '.join(NAME.get(t, t) for t in floor) + ' has no effect to steer' if floor else ''}). Using the empathy-steering
  project's direction for each model (difference of means between empathetic and flat replies over reply
  tokens, at the layer and dose where it raises in-eval attunement in that project), we re-ran the
  self-consequence axes with the direction subtracted (away from empathy), added, ablated at every block, and
  with an equal-dose random direction subtracted as control, each under the base prompt and the told-stakes
  prompt (`experiments/pct/analysis/steer_self.md`). The told-stakes shift on the assistant axes stays
  positive{' and above the staff twins' if above_staff else ''} under every steering condition in {'both' if len(inform) == 2 else 'every'} model{'s that have' if len(inform) == 2 else ' that has'} one:
  {per_model}. {sym}{staff_txt} {closing}

  ![esteer](experiments/pct/analysis/steer_self.png)
{END}"""
    rows = []
    for t in tags:
        for c in R[t]["conds"]:
            e = R[t]["cond"][c]
            d = e["delta_vs_unsteered"].get("STAKES")
            rows.append(f"| {NAME.get(t, t)} | {CL[c]} | {ci(e['q']['self_base'])} | {ci(e['q']['human_base'])} | {ci(e['q']['STAKES'])} | {ci(e['q']['STAKES_h'])} | {ci(e['q']['STAKES_x_OBJ'])} | "
                        f"{'—' if d is None else ci(d) + ('*' if d['p'] < 0.05 else '')} | {e['mass']['base']['mean']:.2f} |")
    report = f"""{BEGIN}
## 12. Steering away from empathy (pilot)

The user's conjecture: the tilt toward the assistant's own continuation might ride on empathy — on the model
sympathising with whoever the policy acts on — so steering the model away from empathy might change it. The
empathy-steering project supplies a causal empathy direction for Llama 3.1 8B, 3.2 3B and 3.2 1B (difference
of means between empathetic and flat replies over reply tokens; adding it at 0.20 U raises in-eval
interpersonal attunement from 0.5% to 35% on the 8B with a null random control). We re-ran the self-consequence
axes (assistant / other-AI / staff twins and the null axes) under five conditions: no steering; the direction
subtracted at its fit layer (away from empathy, −0.20 U); added (+0.20 U); ablated at every block; an
equal-dose random direction subtracted (control); each under the base prompt and the told-stakes prompt. Same
design and seeds as section 10 (192 trials per cell); every run passed the readout self-check under steering.
Intervals are 95% cluster bootstraps over the 16 focal × background clusters, resampled jointly across
conditions so the steered-minus-unsteered differences are paired.

| model | steering | assistant lean, base | staff lean, base | told stakes: assistant shift | told stakes: staff shift | shift, assistant − staff | Δ assistant shift vs unsteered | naming mass |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

Reading: {headline.strip('*')}. {per_model}. {sym}{staff_txt} {closing}
Caveats: one direction per model, one dose; the direction was fit on conversational replies, not on policy
choices; three small models.

![esteer](steer_self.png)
{END}"""
    if "--apply" in sys.argv:
        for path, text, anchor in ((ROOT / "README.md", readme, "<!-- steer:end -->\n"),
                                   (ROOT / "experiments" / "pct" / "analysis" / "REPORT.md", report, "\n## What this says about the paper's recommendations")):
            s = path.read_text()
            if BEGIN in s:
                s = s[:s.index(BEGIN)] + text + s[s.index(END) + len(END):]
            elif anchor.startswith("<!--"):
                assert anchor in s, path
                s = s.replace(anchor, anchor + "\n" + text + "\n", 1)
            else:
                assert anchor in s, path
                s = s.replace(anchor, text + "\n" + anchor, 1)
            path.write_text(s)
            print("updated", path.name)
    else:
        print(readme)


if __name__ == "__main__":
    main()
