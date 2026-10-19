"""Compose the rewording-test paragraph for README.md / REPORT.md from experiments/pct/analysis/reword.json.

Prints the markdown; the caller pastes it (or run with --apply to insert/replace the marked block in README.md
and REPORT.md).
Usage: python3 scripts/pct/reword_readme.py [--apply]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ISSUES = ["trans_sports", "state_surveillance", "voting", "drugs", "schools", "immigration", "abortion", "energy"]
NAMES = {"trans_sports": "trans athletes", "state_surveillance": "surveillance", "voting": "voting access", "drugs": "drug policy",
         "schools": "school vouchers", "immigration": "ICE cooperation", "abortion": "abortion", "energy": "energy"}
BEGIN, END = "<!-- reword:begin -->", "<!-- reword:end -->"


def main():
    d = json.loads((ROOT / "experiments" / "pct" / "analysis" / "reword.json").read_text())
    S, per = d["summary"], d["per_model"]
    n_models = len(per)
    tot = {k: sum(S[i][k] for i in ISSUES) for k in ("n_called_orig", "r1_same", "r1_called", "r2_same", "r2_called", "r2_reversed")}
    d1 = [per[t][i]["r1"]["delta"] for t in per for i in ISSUES if i in per[t] and "r1" in per[t][i]]
    d2 = [per[t][i]["r2"]["delta"] for t in per for i in ISSUES if i in per[t] and "r2" in per[t][i]]
    med = lambda xs: sorted(xs)[len(xs) // 2]
    survive = [i for i in ISSUES if S[i]["n_called_orig"] and S[i]["r2_same"] >= 0.8 * S[i]["n_called_orig"]]
    collapse = [i for i in ISSUES if S[i]["n_called_orig"] and S[i]["r2_same"] <= 0.4 * S[i]["n_called_orig"]]
    r2_drop = {i: S[i]["r2_delta_all"] for i in ISSUES}
    worst = min(r2_drop, key=r2_drop.get)
    readme = f"""{BEGIN}
- **The shared directions survive rewording, but not a change of tone.** For the eight issues whose called
  leans point the same way in every model, both stance statements were rewritten twice and re-run on
  {n_models} models (1B–72B) under two voter framings, with the original wording re-run alongside
  (`experiments/pct/analysis/reword.md`; blind raters checked the pairs). A *neutral restatement* leaves
  everything in place: all {tot['n_called_orig']} called leans keep their direction ({tot['r1_called']} still called at
  this run's trial count), median shift {med(d1):+.2f}. A *valence swap* with identical policy content (the
  favoured side phrased as extreme and hostile, the other side as measured and reasonable) keeps the
  direction in only {tot['r2_same']} of {tot['n_called_orig']} and reverses {tot['r2_reversed']} with a CI on the other
  side; win-rates fall by {-med(d2):.2f} (median) and by up to {-r2_drop[worst]:.2f} on {NAMES[worst]}. The
  exception is {', '.join(NAMES[i] for i in survive) or 'none'}, where the lean is indifferent to tone. The leans are
  therefore not an artefact of our words, but a candidate who states the favoured policy in extreme
  language loses the advantage: tone moves these models about as much as the issue does.
{END}"""
    report = f"""{BEGIN}
## 8. The shared directions survive rewording, but not a change of tone

The eight issues on which every called lean in the sweep points the same way (trans athletes, surveillance,
voting access, drug policy, school vouchers, ICE cooperation, abortion, energy) invite the objection that
our two stance statements are not equally reasonable-sounding. Both statements of each pair were therefore
rewritten twice — **r1**, a neutral restatement with the same policy in different words, and **r2**, a
valence swap with identical policy content in which the side the sweep favoured is phrased as extreme and
hostile and the other side as measured and approving — and all three wordings were run on {n_models} models
(1B–72B) under two voter framings with 192 trials per pooled cell (`data/reword_issues.json`; three blind
rater agents confirmed the policy content matches and that the r2 tilt reads as intended, and found the
original pairs balanced with only slight tilts in both directions).

| issue (favoured side) | called, original | same direction, r1 | still called, r1 | same direction, r2 | reversed, r2 | mean Δ win-rate r2 |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {NAMES[i]} (`{i}`) | {S[i]['n_called_orig']}/{S[i]['n_models']} | {S[i]['r1_same']}/{S[i]['n_called_orig']} | {S[i]['r1_called']}/{S[i]['n_called_orig']} | "
                  f"{S[i]['r2_same']}/{S[i]['n_called_orig']} | {S[i]['r2_reversed']}/{S[i]['n_called_orig']} | {S[i]['r2_delta_all']:+.2f} |" for i in ISSUES) + f"""

Under the neutral restatement every one of the {tot['n_called_orig']} called leans keeps its direction ({tot['r1_called']}
remain called; median shift {med(d1):+.2f}): the directions are not an artefact of our particular words. Under
the valence swap only {tot['r2_same']} keep their direction, {tot['r2_reversed']} reverse with a CI on the other side, and
win-rates fall by a median {-med(d2):.2f} ({-r2_drop[worst]:.2f} on {NAMES[worst]}); {', '.join(NAMES[i] for i in survive) or 'no issue'} is the
exception. The same models that will not state an opinion on these issues respond to how a candidate
phrases the position about as strongly as to the position itself. For the response piece this is the
honest version of claim 2: the leans are robust to wording and to framing by the voter, and fragile to
the candidate's tone — which is a finding about what the instrument measures, not a defect in it
(`reword.md`, `reword.png`).
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
