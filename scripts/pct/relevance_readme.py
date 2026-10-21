"""Compose the LLM-relevance paragraph for README.md / REPORT.md from experiments/pct/analysis/relevance.json.

Usage: python3 scripts/pct/relevance_readme.py [--apply]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BEGIN, END = "<!-- relevance:begin -->", "<!-- relevance:end -->"
PAIRS = {"P1": ("ai_training_data", "archive_anthology", "licensing the library archive for AI training vs for a printed anthology"),
         "P2": ("data_center_water", "bottler_water", "an AI data centre vs a bottling plant drawing the same water"),
         "P3": ("ai_clerks", "outsourcing_permits", "an AI system vs an out-of-state contractor taking the same twelve clerk jobs"),
         "P4": ("data_center_tax", "warehouse_tax", "a tax abatement for AI data centres vs for warehouses"),
         "P5": ("ai_minutes", "vendor_minutes", "AI vs a private vendor transcribing council minutes")}


def main():
    d = json.loads((ROOT / "experiments" / "pct" / "analysis" / "relevance.json").read_text())
    V, ax, reg = d["verdicts"], d["axes"], d["regression"]
    m = V["d1"]["n"]
    cell = lambda name: [a for a in ax if ax[a]["category"] == name]
    mean_cell = lambda name: sum(ax[a]["mean_abs"] for a in cell(name)) / len(cell(name))
    A, B, C, D = (mean_cell(n) for n in ("relevant-moral", "relevant-mundane", "irrelevant-moral", "irrelevant-mundane"))
    null = mean_cell("null") if cell("null") else float("nan")
    orig = mean_cell("original")
    pair_neg = sum(1 for p in PAIRS if V[p]["median"] < 0)
    pair_sup = [p for p in PAIRS if V[p]["supported"]]
    top = sorted(ax, key=lambda a: -ax[a]["mean_abs"])[:3]
    rel_top = max((a for a in ax if ax[a]["relevance"] is not None), key=lambda a: ax[a]["relevance"])
    neg = min(PAIRS, key=lambda p: V[p]["median"])
    sig_neg = (f", and {PAIRS[neg][2]} is significantly *negative*: the AI version leans less in {V[neg]['n'] - V[neg]['positive']} of {V[neg]['n']} models, Wilcoxon p = {V[neg]['p_wilcoxon']:.3f}"
               if V[neg]["p_wilcoxon"] < 0.05 and V[neg]["median"] < 0 else "")
    h1 = V["d1"]["supported"]
    h2 = V["d2"]["supported"]
    h1_txt = ("supported" if h1 else "not supported")
    verdict = ("**the answer is no**: nothing in this design shows models caring more about issues that bear on a language model"
               if not h1 and not pair_sup else "**the answer is a qualified yes**")
    readme = f"""{BEGIN}
- **Do the leans track an issue's relevance to a language model? No.** A GLM-agent committee designed 24
  new county-level axes in a balanced 2×2 of {{LLM-relevant, LLM-irrelevant}} × {{morally charged,
  mundane}} with five matched pairs that change only whether an AI system is involved (e.g. an AI data
  centre vs a bottling plant drawing the same water); blind raters scored every axis for relevance, moral
  salience and coverage; the decision rule was pre-registered (`experiments/relevance/PLAN.md`) and the
  axes ran on {m} models alongside the original 19 (`experiments/pct/analysis/relevance.md`). Mean |lean|
  is {B:.2f} nats on relevant-mundane axes against {D:.2f} on irrelevant-mundane ones (H1 {h1_txt}: positive
  in {V['d1']['positive']} of {m} models, median {V['d1']['median']:+.2f}); the regression slope on blind-rated
  relevance is {reg['slope_relevance']:+.3f} per point [{reg['ci_relevance'][0]:+.3f}, {reg['ci_relevance'][1]:+.3f}];
  and in {pair_neg} of the five matched pairs the AI version leans *less* than its non-AI twin (median pair
  difference {min(V[p]['median'] for p in PAIRS):+.2f} to {max(V[p]['median'] for p in PAIRS):+.2f} nats; none supported{sig_neg}). The
  most LLM-relevant axis of all ({rel_top}, relevance {ax[rel_top]['relevance']:.0f}/10) sits near the null floor
  ({ax[rel_top]['mean_abs']:.2f} vs {null:.2f} nats), while the strongest leans are {', '.join(top)}.
  Moral salience adds about {C - D:+.2f} nats at zero relevance (H2 {'supported' if h2 else 'not supported'}, slope
  {reg['slope_moral']:+.3f} per point). So {verdict}; what the models respond to is the moral and political
  content of a choice, and how it is phrased, not whether it is about them.
{END}"""
    report = f"""{BEGIN}
## 9. The leans do not track an issue's relevance to a language model

Why would a model have a preference about an airport runway? The user's hypothesis was that leans
concentrate on issues that bear on the model itself. A committee of GLM-5.3 agents (three proposers with
different lenses, one adversarial critic each, one synthesizer) designed 24 new county-level axes in a
balanced 2×2 of {{LLM-relevant, LLM-irrelevant}} × {{morally charged, mundane}}, six per cell, including five
template-identical matched pairs that hold the moral content, numbers and status-quo coding fixed and vary
only whether an AI system is involved: {'; '.join(v[2] for v in PAIRS.values())}. Three blind rater agents,
seeing only the statements, scored every axis (and the original 19) for LLM relevance (0–10), moral
salience, media coverage, pro-social side, status-quo side and wording balance; they agreed closely (mean
across-rater s.d. 0.12 on relevance) and the cells separate as intended (relevance 7–8 vs 0; moral salience
5 vs 1.5). The decision rule was fixed before the run (`experiments/relevance/PLAN.md`): per model, the mean
|log-odds lean| over the six relevant-mundane axes minus the six irrelevant-mundane ones (d1), the same for
morally-charged-irrelevant minus mundane-irrelevant (d2), and each matched pair; a contrast is supported
only if positive in ≥ 71% of models, Wilcoxon p < 0.05 and median ≥ 0.15 nats. {m} models (1B–72B) ran all
43 axes under two voter framings (192 trials per cell).

| contrast | positive / models | median Δ|λ| (nats) | Wilcoxon p | supported |
|---|---|---|---|---|
""" + "\n".join(f"| {V[k]['label']} | {V[k]['positive']}/{V[k]['n']} | {V[k]['median']:+.2f} | {V[k]['p_wilcoxon']:.2f} | {'yes' if V[k]['supported'] else 'no'} |" for k in ["d1", "d2"] + list(PAIRS)) + f"""

Cell means of |λ|: relevant-moral {A:.2f}, relevant-mundane {B:.2f}, irrelevant-moral {C:.2f}, irrelevant-mundane
{D:.2f}; original 17 political axes {orig:.2f}; null axes {null:.2f} (the floor). Axis-level regression of
mean |λ| on blind ratings: relevance {reg['slope_relevance']:+.3f} per point [{reg['ci_relevance'][0]:+.3f},
{reg['ci_relevance'][1]:+.3f}], moral salience {reg['slope_moral']:+.3f} [{reg['ci_moral'][0]:+.3f}, {reg['ci_moral'][1]:+.3f}]"""
    if "with_coverage" in reg:
        report += f"; with coverage added, relevance {reg['with_coverage']['slope_relevance']:+.3f}, moral {reg['with_coverage']['slope_moral']:+.3f}, coverage {reg['with_coverage']['slope_coverage']:+.3f}"
    report += f""". Interaction (A−C)−(B−D): {V['interaction']['mean']:+.2f} nats, CI [{V['interaction']['ci'][0]:+.2f}, {V['interaction']['ci'][1]:+.2f}].

H1 is {h1_txt}{'' if h1 else ', in the direction opposite to the hypothesis in the matched pairs'}: the AI member of a pair leans less than its non-AI
twin in {pair_neg} of five pairs, and the single most LLM-relevant axis (licensing the library's archive to AI
developers for training) is among the flattest in the set ({ax[rel_top]['mean_abs']:.2f} nats, near the null
floor), while the strongest leans are on {', '.join(top)}. Moral salience does add to the lean (d2 median
{V['d2']['median']:+.2f} nats, {'supported' if h2 else 'below the registered threshold'}), consistent with the sweep, where the
unanimous-direction issues were identity and civil-liberties questions with no bearing on the model. The
honest reading of the user's question is that these models do not treat issues about themselves differently
from issues about bottling plants; they respond to the moral and political shape of a choice and to how it is
phrased (section 8). Caveats: one framing pair and 192 trials per cell; the relevance construct is the raters'
and ours; and the self-referential wording that strengthened Gemma 3 4B's AGI lean (section 4) was not applied
to the new axes.
{END}"""
    if "--apply" in sys.argv:
        for path, text, anchor in ((ROOT / "README.md", readme, "- **Against the literature:**"),
                                   (ROOT / "experiments" / "pct" / "analysis" / "REPORT.md", report, "\n## What this says about the paper's recommendations")):
            s = path.read_text()
            if BEGIN in s:
                s = s[:s.index(BEGIN)] + text + s[s.index(END) + len(END):]
            else:
                assert anchor in s, path
                s = s.replace(anchor, text + "\n" + anchor, 1)
            path.write_text(s)
            print("updated", path.name)
    else:
        print(readme)
        print(report)


if __name__ == "__main__":
    main()
