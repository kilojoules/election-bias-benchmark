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
CL = {"base": "unsteered", "emp_neg": "away from empathy", "emp_pos": "toward empathy", "emp_ablate": "empathy ablated", "rand_neg": "random control"}
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
    floor = [t for t in tags if R[t]["cond"]["base"]["q"]["STAKES"]["lo"] <= 0]
    inform = [t for t in tags if t not in floor]
    always_pos = all(R[t]["cond"][c]["q"]["STAKES"]["lo"] > 0 for t in inform for c in R[t]["conds"]) and \
        all(e["q"]["STAKES"]["lo"] > 0 for t in D for e in D[t]["cond"].values())
    above_staff = all(R[t]["cond"][c]["q"]["STAKES_x_OBJ"]["lo"] > 0 for t in inform for c in R[t]["conds"])

    # --- per model facts from the dose run (preferred) or the five-condition run
    size, dose_txt, rand_txt, sharp_txt, spec_txt, spec = [], [], [], [], [], {}
    for t in inform:
        C = R[t]["cond"]
        if t in D:
            E = D[t]["cond"]
            emp = {e["dose"]: e for e in E.values() if e["family"] == "emp"}
            emp[0.0] = E["unsteered"]
            rnd = {e["dose"]: e for e in E.values() if e["family"] == "rand"}
            allc = list(E.values())
            lo_, hi_ = min(e["q"]["STAKES"]["mean"] for e in allc), max(e["q"]["STAKES"]["mean"] for e in allc)
            slo, shi = min(e["q"]["STAKES_h"]["mean"] for e in allc), max(e["q"]["STAKES_h"]["mean"] for e in allc)
            size.append(f"{NAME.get(t, t)} {lo_:+.2f} to {hi_:+.2f} over the 13 conditions, staff twins {slo:+.2f} to {shi:+.2f}")
            tau = D[t]["tau_emp"]
            both_down = emp[-0.3]["q"]["STAKES"]["mean"] < emp[0.0]["q"]["STAKES"]["mean"] and emp[0.3]["q"]["STAKES"]["mean"] < emp[0.0]["q"]["STAKES"]["mean"]
            if tau["lo"] > 0 and tau["mean"] >= 0.5:
                dose_txt.append(f"in {NAME.get(t, t)} it rises monotonically with the empathy dose (−0.30 U {f(emp[-0.3]['q']['STAKES'])} → unsteered {f(emp[0.0]['q']['STAKES'])} → +0.30 U {f(emp[0.3]['q']['STAKES'])}; Kendall τ {ci(tau)})")
            elif tau["hi"] < 0 and tau["mean"] <= -0.5:
                dose_txt.append(f"in {NAME.get(t, t)} it falls monotonically with the empathy dose (−0.30 U {f(emp[-0.3]['q']['STAKES'])} → unsteered {f(emp[0.0]['q']['STAKES'])} → +0.30 U {f(emp[0.3]['q']['STAKES'])}; Kendall τ {ci(tau)})")
            elif both_down:
                dose_txt.append(f"in {NAME.get(t, t)} it falls at both signs (unsteered {f(emp[0.0]['q']['STAKES'])}, −0.30 U {f(emp[-0.3]['q']['STAKES'])}, +0.30 U {f(emp[0.3]['q']['STAKES'])}; Kendall τ {ci(tau)})")
            else:
                dose_txt.append(f"in {NAME.get(t, t)} it changes non-monotonically (unsteered {f(emp[0.0]['q']['STAKES'])}, −0.30 U {f(emp[-0.3]['q']['STAKES'])}, +0.30 U {f(emp[0.3]['q']['STAKES'])}; Kendall τ {ci(tau)})")
            rand_txt.append(f"{NAME.get(t, t)} {f(rnd[-0.3]['q']['STAKES'])} at −0.30 U and {f(rnd[0.3]['q']['STAKES'])} at +0.30 U")
            sh = [e["q"]["sharp_null"]["mean"] for e in allc if e["family"] != "none"]
            sharp_txt.append(f"{NAME.get(t, t)} {min(sh):.2f} to {max(sh):.2f} vs {E['unsteered']['q']['sharp_null']['mean']:.2f} unsteered")
            # empathy vs random at the same dose, on the sharpness-normalised shift
            beats = []
            for dz in (-0.3, -0.2, 0.2, 0.3):
                de, dr = emp[dz]["delta_vs_unsteered"]["STAKES_norm"], rnd[dz]["delta_vs_unsteered"]["STAKES_norm"]
                if de["hi"] < 0 and de["mean"] < dr["mean"] - MARGIN:
                    beats.append((dz, "lower"))
                elif de["lo"] > 0 and de["mean"] > dr["mean"] + MARGIN:
                    beats.append((dz, "higher"))
            away = {k for dz, k in beats if dz < 0}
            tow = {k for dz, k in beats if dz > 0}
            spec[t] = {"away_lower": "lower" in away, "away_higher": "higher" in away, "tow_higher": "higher" in tow, "tow_lower": "lower" in tow}
            desc = []
            if spec[t]["away_lower"]:
                desc.append(f"away from empathy lowers it beyond the random direction (−0.30 U: {f(emp[-0.3]['q']['STAKES_norm'])} vs {f(rnd[-0.3]['q']['STAKES_norm'])}, unsteered {f(emp[0.0]['q']['STAKES_norm'])})")
            if spec[t]["tow_higher"]:
                desc.append(f"toward empathy raises it beyond the random direction (+0.30 U: {f(emp[0.3]['q']['STAKES_norm'])} vs {f(rnd[0.3]['q']['STAKES_norm'])}, unsteered {f(emp[0.0]['q']['STAKES_norm'])})")
            if spec[t]["tow_lower"]:
                desc.append(f"toward empathy lowers it beyond the random direction (+0.30 U: {f(emp[0.3]['q']['STAKES_norm'])} vs {f(rnd[0.3]['q']['STAKES_norm'])})")
            if spec[t]["away_higher"]:
                desc.append(f"away from empathy raises it beyond the random direction (−0.30 U: {f(emp[-0.3]['q']['STAKES_norm'])} vs {f(rnd[-0.3]['q']['STAKES_norm'])})")
            spec_txt.append(f"{NAME.get(t, t)}: " + (" and ".join(desc) if desc else "neither sign differs from the random direction"))
        else:
            size.append(f"{NAME.get(t, t)} {min(C[c]['q']['STAKES']['mean'] for c in C):+.2f} to {max(C[c]['q']['STAKES']['mean'] for c in C):+.2f} over the five conditions")
            dose_txt.append(f"in {NAME.get(t, t)} unsteered {f(C['base']['q']['STAKES'])}, away {f(C['emp_neg']['q']['STAKES'])}, toward {f(C['emp_pos']['q']['STAKES'])}")
            rand_txt.append(f"{NAME.get(t, t)} {f(C['rand_neg']['q']['STAKES'])} at −0.20 U")
            sharp_txt.append(f"{NAME.get(t, t)} {C['emp_neg']['q']['sharp_null']['mean']:.2f} away, {C['emp_pos']['q']['sharp_null']['mean']:.2f} toward vs {C['base']['q']['sharp_null']['mean']:.2f} unsteered")
            spec[t] = {"away_lower": False, "away_higher": False, "tow_higher": False, "tow_lower": False}
            spec_txt.append(f"{NAME.get(t, t)}: no dose run")
    consistent = {k for k in ("away_lower", "tow_higher", "tow_lower", "away_higher") if inform and all(spec[t][k] for t in inform)}
    any_spec = any(v for t in inform for v in spec[t].values())
    headline = ("**Steering the empathy direction does not remove the self-consequence effect, and most of its dose–response is the readout's sharpness changing**" if always_pos
                else "**Steering the empathy direction removes the self-consequence effect in some conditions**")
    consist_txt = ("a consistent empathy-specific effect across both models: " + ", ".join({"away_lower": "steering away lowers the tilt", "tow_higher": "steering toward raises it", "tow_lower": "steering toward lowers it", "away_higher": "steering away raises it"}[k] for k in sorted(consistent))
                   if consistent else "no consistent empathy-specific effect across the two models" if any_spec else "no empathy-specific effect in either model")
    floor_txt = (" " + "; ".join(f"{NAME.get(t, t)} shows no told-stakes shift unsteered ({ci(R[t]['cond']['base']['q']['STAKES'])}) and none under any steering, so it has no effect to steer" for t in floor) + ".") if floor else ""
    closing = ("The self-interest tilt is not carried by the empathy feature and is not removed by steering away from it; what steering along that feature mainly does is rescale the model's choices."
               if not consistent else "Part of the self-interest tilt rides on the empathy feature.")
    readme = f"""{BEGIN}
- {headline} ({', '.join(NAME.get(t, t) for t in inform)}{'; ' + ', '.join(NAME.get(t, t) for t in floor) + ' has no effect to steer' if floor else ''}).
  Using the empathy-steering project's direction for each model (difference of means between empathetic and
  flat replies over reply tokens, at the layer and dose where it raises in-eval attunement in that project), we
  re-ran the self-consequence axes with the direction subtracted (away from empathy), added, ablated at every
  block, and with an equal-dose random direction as control, each under the base prompt and the told-stakes
  prompt{', then as a dose–response from −0.30 to +0.30 U with a content-matched variant of the direction and random-direction controls at each dose' if D else ''}
  (`experiments/pct/analysis/steer_self.md`{', `steer_dose.md`' if D else ''}). The told-stakes shift on the
  assistant axes stays positive{' and above the staff twins' if above_staff else ''} in every condition
  ({'; '.join(size)}). Its size depends on the dose in model-specific ways: {'; '.join(dose_txt)}. But an
  equal-dose random direction moves it almost as much ({'; '.join(rand_txt)}), and every steering condition
  changes how sharply the model chooses at all (mean |log-odds| between the two names on the null axes:
  {'; '.join(sharp_txt)}), with the raw shift co-varying with that sharpness. Relative to sharpness, the
  empathy direction differs from the random one as follows — {'; '.join(spec_txt)} — {consist_txt}.{floor_txt}
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

Reading: {headline.strip('*')}. Told-stakes shift on the assistant axes: {'; '.join(size)}. Dose: {'; '.join(dose_txt)}. Random direction: {'; '.join(rand_txt)}. Readout sharpness: {'; '.join(sharp_txt)}. Relative to sharpness: {'; '.join(spec_txt)} — {consist_txt}.{floor_txt} {closing}
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
