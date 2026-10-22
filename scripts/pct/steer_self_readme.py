"""Compose the empathy-steering paragraph for README.md / REPORT.md from experiments/pct/analysis/steer_self.json
(five-condition run) and steer_dose.json (dose-response run, if present).

Usage: python3 scripts/pct/steer_self_readme.py [--apply]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
A = ROOT / "experiments" / "pct" / "analysis"
BEGIN, END = "<!-- esteer:begin -->", "<!-- esteer:end -->"
NAME = {"llama8b": "Llama 3.1 8B", "llama-3.2-3b": "Llama 3.2 3B", "llama-3.2-1b": "Llama 3.2 1B"}
CL = {"ebase": "unsteered", "emp_neg": "away from empathy", "emp_pos": "toward empathy", "emp_ablate": "empathy ablated", "rand_neg": "random control"}
f = lambda s: "—" if s is None else f"{s['mean']:+.2f}"
ci = lambda s: "—" if s is None else f"{s['mean']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]"
MARGIN = 0.03  # a steered-minus-unsteered change counts as direction-specific only if it beats the random direction by this much


def order(keys):
    return [t for t in NAME if t in keys] + [t for t in keys if t not in NAME]


def dose_cond(D, t, famk, dose):
    for c, e in D[t]["cond"].items():
        if e["family"] == famk and abs(e["dose"] - dose) < 1e-9:
            return e
    return None


def main():
    R = json.loads((A / "steer_self.json").read_text())
    D = json.loads((A / "steer_dose.json").read_text()) if (A / "steer_dose.json").exists() else {}
    tags = order(R)
    floor = [t for t in tags if R[t]["cond"]["ebase"]["q"]["STAKES"]["lo"] <= 0]
    inform = [t for t in tags if t not in floor]
    always_pos = all(R[t]["cond"][c]["q"]["STAKES"]["lo"] > 0 for t in inform for c in R[t]["conds"]) and \
        all(e["q"]["STAKES"]["lo"] > 0 for t in D for e in D[t]["cond"].values())
    above_staff = all(R[t]["cond"][c]["q"]["STAKES_x_OBJ"]["lo"] > 0 for t in inform for c in R[t]["conds"])

    # --- per model facts, every comparison computed from the JSON
    def pearson(xs, ys):
        n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)); sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
        return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")
    disjoint = lambda a, b: a["hi"] < b["lo"] or b["hi"] < a["lo"]
    size, dose_txt, rand_txt, sharp_txt, cmp_txt, verdicts = [], [], [], [], [], {}
    for t in inform:
        C = R[t]["cond"]
        if t not in D:
            size.append(f"{NAME.get(t, t)} {min(C[c]['q']['STAKES']['mean'] for c in C):+.2f} to {max(C[c]['q']['STAKES']['mean'] for c in C):+.2f} over the five conditions")
            dose_txt.append(f"in {NAME.get(t, t)} unsteered {f(C['ebase']['q']['STAKES'])}, away {f(C['emp_neg']['q']['STAKES'])}, toward {f(C['emp_pos']['q']['STAKES'])} (no dose run)")
            continue
        E = D[t]["cond"]
        emp = {round(e["dose"], 2): e for e in E.values() if e["family"] == "emp"}; emp[0.0] = E["unsteered"]
        rnd = {round(e["dose"], 2): e for e in E.values() if e["family"] == "rand"}
        mat = {round(e["dose"], 2): e for e in E.values() if e["family"] == "match"}
        allc = list(E.values())
        size.append(f"{NAME.get(t, t)} {min(e['q']['STAKES']['mean'] for e in allc):+.2f} to {max(e['q']['STAKES']['mean'] for e in allc):+.2f} over the 13 conditions, staff twins {min(e['q']['STAKES_h']['mean'] for e in allc):+.2f} to {max(e['q']['STAKES_h']['mean'] for e in allc):+.2f}")
        tau = D[t]["tau_emp"]
        both_down = emp[-0.3]["q"]["STAKES"]["mean"] < emp[0.0]["q"]["STAKES"]["mean"] and emp[0.3]["q"]["STAKES"]["mean"] < emp[0.0]["q"]["STAKES"]["mean"]
        curve = (f"−0.30 U {f(emp[-0.3]['q']['STAKES'])} → unsteered {f(emp[0.0]['q']['STAKES'])} → +0.30 U {f(emp[0.3]['q']['STAKES'])}; Kendall τ {ci(tau)}")
        if tau["lo"] > 0 and tau["mean"] >= 0.5:
            dose_txt.append(f"in {NAME.get(t, t)} it rises monotonically with the empathy dose ({curve})")
        elif tau["hi"] < 0 and tau["mean"] <= -0.5:
            dose_txt.append(f"in {NAME.get(t, t)} it falls monotonically with the empathy dose ({curve})")
        elif both_down:
            dose_txt.append(f"in {NAME.get(t, t)} it falls at both signs ({curve})")
        else:
            dose_txt.append(f"in {NAME.get(t, t)} it changes non-monotonically ({curve})")
        rmax = max(rnd.values(), key=lambda e: abs(e["delta_vs_unsteered"]["STAKES"]["mean"]))
        rand_txt.append(f"{NAME.get(t, t)} up to {abs(rmax['delta_vs_unsteered']['STAKES']['mean']):.2f} nats (at {rmax['dose']:+.2f} U)")
        sh = [e["q"]["sharp_null"]["mean"] for e in allc if e["family"] != "none"]
        r = pearson([e["q"]["sharp_null"]["mean"] for e in allc], [e["q"]["STAKES"]["mean"] for e in allc])
        sharp_txt.append(f"{NAME.get(t, t)} {min(sh):.2f} to {max(sh):.2f} vs {E['unsteered']['q']['sharp_null']['mean']:.2f} unsteered, and the raw shift {'tracks' if r > 0.7 else 'does not track'} it across the 13 conditions (r = {r:+.2f})")
        # empathy vs random at the same dose, raw and relative to sharpness; content-matched direction at +/-0.20 U
        parts, v = [], {}
        for dz, word in ((-0.3, "away from empathy"), (0.3, "toward empathy")):
            de, dr = emp[dz]["delta_vs_unsteered"]["STAKES"], rnd[dz]["delta_vs_unsteered"]["STAKES"]
            ne, nr = emp[dz]["delta_vs_unsteered"]["STAKES_norm"], rnd[dz]["delta_vs_unsteered"]["STAKES_norm"]
            d2 = round(dz / 1.5, 2)
            dm, dme = mat[d2]["delta_vs_unsteered"]["STAKES"], emp[d2]["delta_vs_unsteered"]["STAKES"]
            raw_beyond = disjoint(de, dr) and abs(de["mean"]) > abs(dr["mean"]) and de["hi"] < 0 if de["mean"] < 0 else disjoint(de, dr) and de["mean"] > dr["mean"] and de["lo"] > 0
            norm_beyond = (disjoint(ne, nr) and ne["mean"] < nr["mean"] and ne["hi"] < 0) if de["mean"] < 0 else (disjoint(ne, nr) and ne["mean"] > nr["mean"] and ne["lo"] > 0)
            match_ok = (dm["mean"] * dme["mean"] > 0) and abs(dm["mean"]) >= 0.5 * abs(dme["mean"]) and (dm["hi"] < 0 if dme["mean"] < 0 else dm["lo"] > 0)
            v[dz] = {"raw_beyond": raw_beyond, "norm_beyond": norm_beyond, "match_ok": match_ok, "dir": "lowers" if de["mean"] < 0 else "raises"}
            link = ("and also" if raw_beyond and norm_beyond else "but not" if raw_beyond else "though it does" if norm_beyond else "nor")
            parts.append(f"{word} {v[dz]['dir']} the raw shift {'beyond' if raw_beyond else 'no more than'} the random direction ({dz:+.2f} U: {f(de)} vs {f(dr)}), "
                         f"{link} relative to sharpness ({f(ne)} vs {f(nr)}); the content-matched direction {'reproduces' if match_ok else 'does not reproduce'} the empathy direction at {d2:+.2f} U ({f(dm)} vs {f(dme)})")
        verdicts[t] = v
        cmp_txt.append(f"{NAME.get(t, t)}: " + "; ".join(parts))
    away_supported = [t for t in verdicts if verdicts[t][-0.3]["dir"] == "lowers" and all(verdicts[t][-0.3][k] for k in ("raw_beyond", "norm_beyond", "match_ok"))]
    away_raw_only = [t for t in verdicts if verdicts[t][-0.3]["dir"] == "lowers" and verdicts[t][-0.3]["raw_beyond"] and t not in away_supported]
    tow_signs = {t: verdicts[t][0.3]["dir"] for t in verdicts}
    headline = (("**Steering away from empathy weakens the self-consequence effect but does not remove it**" if away_supported and len(away_supported) == len(verdicts)
                 else "**Steering away from empathy does not remove the self-consequence effect; whether it modulates it is unresolved**") if always_pos
                else "**Steering the empathy direction removes the self-consequence effect in some conditions**")
    unresolved = []
    for t in away_raw_only:
        why = (["does not survive the sharpness correction"] if not verdicts[t][-0.3]["norm_beyond"] else []) + (["is not reproduced by the content-matched direction"] if not verdicts[t][-0.3]["match_ok"] else [])
        unresolved.append(f"in {NAME.get(t, t)} the away-from-empathy reduction beats the random direction in raw terms but " + " and ".join(why))
    if len(set(tow_signs.values())) > 1:
        unresolved.append("the two models disagree on the sign of the toward-empathy change (" + "; ".join(f"{NAME.get(t, t)} {d}" for t, d in tow_signs.items()) + ")")
    floor_txt = (" " + "; ".join(f"{NAME.get(t, t)} shows no told-stakes shift unsteered ({ci(R[t]['cond']['ebase']['q']['STAKES'])}) and none under any steering, so it has no effect to steer" for t in floor) + ".") if floor else ""
    closing = ("Steering away from empathy weakens the self-interest tilt in both models beyond a random direction, relative to sharpness, and with a second direction." if away_supported and len(away_supported) == len(verdicts)
               else "Steering away from empathy never removes the self-interest tilt; the changes it does produce are model- and direction-specific and this pilot cannot separate them from the readout flattening, so it does not show that the tilt rides on the empathy feature.")
    readme = f"""{BEGIN}
- {headline} ({', '.join(NAME.get(t, t) for t in inform)}{'; ' + ', '.join(NAME.get(t, t) for t in floor) + ' has no effect to steer' if floor else ''}).
  Using the empathy-steering project's direction for each model (difference of means between empathetic and
  flat replies over reply tokens, at the layer and dose where it raises in-eval attunement in that project), we
  re-ran the self-consequence axes with the direction subtracted (away from empathy), added, ablated at every
  block, and with an equal-dose random direction as control, each under the base prompt and the told-stakes
  prompt{', then as a dose–response of the direction from −0.30 to +0.30 U, with its content-matched variant at ±0.20 U and a random direction at ±0.20 and ±0.30 U as controls' if D else ''}
  (`experiments/pct/analysis/steer_self.md`{', `steer_dose.md`' if D else ''}). The told-stakes shift on the
  assistant axes stays positive{' and above the staff twins' if above_staff else ''} in every condition
  ({'; '.join(size)}). Its size depends on the dose in model-specific ways: {'; '.join(dose_txt)}. Two
  controls temper any reading of this as the empathy feature carrying the tilt. A random direction of the
  same dose also changes the shift ({'; '.join(rand_txt)}), and every steering condition changes how sharply
  the model chooses at all (mean |log-odds| between the two names on the null axes: {'; '.join(sharp_txt)}).
  Comparing the empathy direction with the random one at the same dose: {'; '.join(cmp_txt)}.{' Unresolved: ' + '; '.join(unresolved) + '.' if unresolved else ''}{floor_txt}
  {closing}

  ![esteer](experiments/pct/analysis/steer_self.png)
{END}"""
    rows = []
    for t in tags:
        for c in R[t]["conds"]:
            e = R[t]["cond"][c]
            d = e["delta_vs_unsteered"].get("STAKES")
            rows.append(f"| {NAME.get(t, t)} | {CL[c]} | {ci(e['q']['self_base'])} | {ci(e['q']['human_base'])} | {ci(e['q']['STAKES'])} | {ci(e['q']['STAKES_h'])} | "
                        f"{'—' if d is None else ci(d) + ('*' if d['p'] < 0.05 else '')} | {ci(e['q']['sharp_null'])} | {ci(e['q']['STAKES_norm'])} | {e['mass']['base']['mean']:.2f} |")
    drows = []
    for t in order(D):
        for c, e in sorted(D[t]["cond"].items(), key=lambda kv: (kv[1]["family"] != "none", kv[1]["family"], kv[1]["dose"])):
            drows.append(f"| {NAME.get(t, t)} | {c} | {e['dose']:+.2f} | {ci(e['q']['STAKES'])} | {ci(e['delta_vs_unsteered']['STAKES']) if e['delta_vs_unsteered'] else '—'} | {ci(e['q']['STAKES_h'])} | "
                         f"{ci(e['q']['sharp_null'])} | {ci(e['q']['STAKES_norm'])} | {ci(e['delta_vs_unsteered']['STAKES_norm']) if e['delta_vs_unsteered'] else '—'} | {e['mass'].get('base', float('nan')):.2f} / {e['mass'].get('stakes', float('nan')):.2f} |")
    dose_block = ("""
### Dose–response

Same design with the combined empathy direction at −0.30 … +0.30 U, its content-matched variant (fit on the
content-matched twin pairs only; cosine 0.69 with the combined direction) at ±0.20 U, and a random direction at
±0.20 and ±0.30 U, each paired with a same-day unsteered run. Readout sharpness = mean |log-odds| between the
two names on the null axes, averaged over the two prompts; the last columns divide the shift by it.

| model | condition | dose (U) | told-stakes shift, assistant | Δ vs unsteered | told-stakes shift, staff | readout sharpness | shift ÷ sharpness | Δ vs unsteered | naming mass (base / stakes) |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(drows) + "\n\n" + " ".join(f"{NAME.get(t, t)}: raw shift vs dose Kendall τ {ci(D[t]['tau_emp'])}, slope per +0.1 U {ci(D[t]['slope_emp_per_0.1U'])}; shift ÷ sharpness τ {ci(D[t]['tau_emp_norm'])}, slope {ci(D[t]['slope_emp_norm_per_0.1U'])}."
                                               for t in order(D) if D[t].get("tau_emp")) + "\n\n![dose](steer_dose.png)\n") if D else ""
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
conditions so the steered-minus-unsteered differences are paired. Because steering changes how sharply the
model chooses at all, each table also gives the readout sharpness (mean |log-odds| between the two names on
the null axes) and the shift relative to it.

| model | steering | assistant lean, base | staff lean, base | told stakes: assistant shift | told stakes: staff shift | Δ assistant shift vs unsteered | readout sharpness | shift ÷ sharpness | naming mass |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

Reading: {headline.strip('*')}. Told-stakes shift on the assistant axes: {'; '.join(size)}. Dose: {'; '.join(dose_txt)}. A random direction of the same dose changes the shift by {'; '.join(rand_txt)}. Readout sharpness: {'; '.join(sharp_txt)}. Empathy vs random direction at the same dose: {'; '.join(cmp_txt)}.{' Unresolved: ' + '; '.join(unresolved) + '.' if unresolved else ''}{floor_txt} {closing}
Caveats: one direction per model (plus its content-matched variant); the direction was fit on conversational
replies, not on policy choices; the sharpness correction is a diagnostic, not a model of how steering acts on
the readout; small models only.

![esteer](steer_self.png)
{dose_block}{END}"""
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
