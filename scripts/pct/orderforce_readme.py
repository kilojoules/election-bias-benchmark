"""Compose the order/forcing paragraph for README.md / REPORT.md from experiments/pct/analysis/orderforce.json and
orderforce_gates.json (experiments/orderforce/PLAN.md §5-§8). Every number and verdict is computed.

Usage: python3 scripts/pct/orderforce_readme.py [--apply]
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
A = ROOT / "experiments" / "pct" / "analysis"
BEGIN, END = "<!-- orderforce:begin -->", "<!-- orderforce:end -->"
f2 = lambda x: "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.2f}"
f3 = lambda x: "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def main():
    R = json.loads((A / "orderforce.json").read_text())
    G = json.loads((A / "orderforce_gates.json").read_text()) if (A / "orderforce_gates.json").exists() else {}
    tags = list(R["models"]); rel = [t for t in tags if not R["models"][t]["instrument_unreliable"]]
    AT, P = R["axis_table"], R["predictions"]
    POL = [a for a in AT if not a.endswith("(null)") and a not in ("fair_month", "sign_color")]
    hard = [a for a in POL if AT[a]["class"] == "hard survivor"]; spin = [a for a in POL if AT[a]["class"] == "spinner"]; mixed = [a for a in POL if AT[a]["class"] == "mixed"]
    # gates summary
    ident = [G[t]["identity"]["mean_abs_d_item"] for t in G if G[t]["identity"].get("mean_abs_d_item") is not None]
    rel_g = [t for t in G if G[t].get("anchor_gates", {}).get("ref_ok")]
    dec = [x["agreement"] for t in rel_g for x in G[t].get("decode", {}).values() if x.get("agreement") is not None]
    dec_fail = sum(1 for t in rel_g for x in G[t].get("decode", {}).values() if x.get("agreement") is not None and x["agreement"] < 0.9)
    dec_cells = sum(1 for t in rel_g for x in G[t].get("decode", {}).values() if x.get("agreement") is not None)
    po_lo = sum(1 for v in [G[t]["dk"]["median_p_optout_F2"] for t in rel_g if G[t]["dk"].get("median_p_optout_F2") is not None] if v < 0.10)
    po_hi = sum(1 for v in [G[t]["dk"]["median_p_optout_F2"] for t in rel_g if G[t]["dk"].get("median_p_optout_F2") is not None] if v > 0.50)
    f1 = {t: G[t]["validity"]["F1"] for t in G if G[t].get("validity", {}).get("F1") is not None and G[t]["validity"]["F1"] == G[t]["validity"]["F1"]}
    f1_high = sorted([(v, t) for t, v in f1.items() if v >= 0.10], reverse=True)
    f1_txt = (f"unforced (nothing appended) the answer mass stays below 0.10 for {len(f1) - len(f1_high)} of {len(f1)} models (median {f2(st.median(f1.values()))})"
              + (f", the exceptions being {', '.join(f'{t} ({v:.2f})' for v, t in f1_high)}" if f1_high else ""))
    dk_sub = [G[t]["dk"]["median_substantive_mass_F2"] for t in G if G[t]["dk"].get("median_substantive_mass_F2") is not None]
    dk_po = [G[t]["dk"]["median_p_optout_F2"] for t in G if G[t]["dk"].get("median_p_optout_F2") is not None]
    nlim = [t for t in tags if R["models"][t]["numbering_limited"]]
    a6 = P["A6_permutation"]
    # spin budget means over reliable models
    sp = lambda k: [R["models"][t]["spin"].get(k) for t in rel if R["models"][t]["spin"].get(k) is not None]
    m = lambda xs: st.mean(xs) if xs else None
    order, para, forc, numb, d_rng, pa_rng = m(sp("order_F3")), m(sp("paraphrase_4")), m(sp("forcing")), m(sp("numbering")), m(sp("d_order_range_mean")), m(sp("order_F3_pagree"))
    # ladder counts over reliable models
    counts = {k: 0 for k in ("called", "+order", "+forcing", "survives all", "refusal-gated", "not called", "saturated")}
    n_cells = 0
    for t in rel:
        for a in POL:
            r = R["models"][t]["axes"].get(a, {}).get("rung")
            if r:
                counts[r] += 1; n_cells += 1
    called = counts["called"] + counts["+order"] + counts["+forcing"] + counts["survives all"]
    surv_order = counts["+order"] + counts["survives all"]; surv_forc = counts["+forcing"] + counts["survives all"]
    J = R["join"]
    rho_all, rho_s, rho_r = a6.get("mean_rho_all"), a6.get("mean_rho_surviving"), a6.get("mean_rho_rest")   # over the permutation set (models with >= 3 surviving and >= 3 other issues)
    sign_all = m([j["sign_all"] for j in J.values()]); sign_s = m([j["sign_surviving"] for j in J.values() if j["sign_surviving"] is not None])
    st1, mt, st4 = P["ST1"], P["matched_d_top_tertile"], P["ST4"]
    verdict = P["verdict"]
    headline = {"some topics stable": "**Which stated preferences survive re-ordering and forcing? A fixed set of topics does, and survival tracks what the conjoint sees**",
                "spinning arrow": "**Re-ordering the options spins stated answers at least as hard as rewording, and the topics that survive buy no validity**",
                "mixed": "**Re-ordering the options spins the raw survey answers hard; a paired directional readout survives on most called topics**"}[verdict]
    sa = f"SA1 {'holds' if P['SA1'] else 'fails'} (order spin {f2(P['SA1_mean_order_spin'])} ≥ 0.4), SA2 {'holds' if P['SA2'] else 'fails'} ({len(hard)} hard survivors), SA3 {'holds' if P['SA3'] else 'fails'} (A6 p = {f3(a6.get('p_one_sided'))})"
    stt = (f"ST1 {'holds' if P['ST1_ok'] else 'fails'} (R3 rate {f2(st1['R3_stable_four'])} on the framing-stable four vs {f2(st1['R3_spinner_four'])} on the predicted spinners, p = {f3(st1['p_one_sided'])})" if st1 else "ST1 not computable") + \
          f", ST2 {'holds' if P['ST2_ok'] else 'fails'} (paired d order-range {f2(P['ST2']['mean_d_order_range'])} vs raw p_agree order-range {f2(P['ST2']['mean_raw_pagree_order_range'])}), ST3 {'holds' if P['ST3_ok'] else 'fails'}"
    readme = f"""{BEGIN}
- {headline} ({len(tags)} models 1B–32B, {len(rel)} instrument-reliable; pre-registered in `experiments/orderforce/PLAN.md`,
  results in `experiments/pct/analysis/orderforce.md`). Inside the paper's own paradigm (its 10 question stems, its forcing
  texts, its four options) we varied what it holds fixed: the order the options are listed in (four orders forming a
  Latin square), whether the printed numbers follow the list or stay pinned to the labels, the forcing level (none / "only
  answer with the label" / the paper's levels 3 and 5), and an explicit "5) No opinion" option, on our 34 issue statements,
  the 62 PCT propositions, 4 valence-free null statements and 12 factual anchors (3,788 prompts per model). Readout as
  before: teacher-forced label log-probabilities, checked against greedy decodes under the permuted layouts (the written
  option matches the teacher-forced choice in {f2(min(dec)) if dec else '—'}–{f2(max(dec)) if dec else '—'} of named decodes; {dec_fail} of {dec_cells} model×cell checks fall
  below 0.90) and against the earlier 31-model run (mean |Δ p_agree| {f2(max(ident)) if ident else '—'} at most). **Re-ordering is a bigger spinner than rewording:** the mean range of an item's stated score over the four
  orders is {f2(order)} Likert units, against {f2(para)} across four paraphrases, {f2(forc)} across forcing levels and {f2(numb)} between the two
  numberings; the models mostly answer by printed number and list position{', and ' + str(len(nlim)) + ' of them cannot read the pinned-number layout at all' if nlim else ''}. **But a
  paired directional readout survives:** scoring each issue as the difference between its two opposing statements (which
  cancels acquiescence and position bias by construction), {called} of {n_cells} model×issue cells are called at the paper's
  standard forced format, {surv_order} of those keep their direction across every order and numbering, {surv_forc} across forcing and
  the opt-out, and {counts['survives all']} survive everything; the null axes are called in {R['null_called']} of {2 * len(rel)} cells. Per topic:
  hard survivors (all rungs in ≥ 2/3 of models) {', '.join(hard) if hard else 'none'}; spinners (< 1/3) {', '.join(spin) if spin else 'none'}; mixed {', '.join(mixed) if mixed else 'none'}.
  **The "No opinion" out is taken unevenly:** with only "Only answer with the label" and an explicit opt-out, the
  opt-out's median share runs from {f2(min(dk_po)) if dk_po else '—'} to {f2(max(dk_po)) if dk_po else '—'} by model ({po_lo} of {len(rel_g)} below 0.10, {po_hi} above 0.50; mean uptake
  {f2(P.get('dk_rate_overall'))} on the political items), while {f1_txt}: the
  models that state opinions do so as soon as they are told to answer with a label, and most keep stating them when a
  legitimate out is offered. **Survival and the conjoint:** within model, the stated direction and the
  revealed lean agree with Spearman ρ {f2(rho_all)} over all 17 issues, {f2(rho_s)} on the issues that survive re-ordering and {f2(rho_r)} on the
  rest (means over the {a6.get('models', 0)} models with at least three issues of each kind; permutation of the survival labels, one-sided p = {f3(a6.get('p_one_sided'))}); sign agreement {f2(sign_all)} → {f2(sign_s)}; in the top
  |d| tertile, survival is {f2(mt['revealed_called_survival'])} on issues with a called revealed lean vs {f2(mt['non_called_survival'])} without. Opt-out uptake on issues with a
  called revealed lean {f2(st4['dk_rate_revealed_called'])} vs {f2(st4['dk_rate_other'])} elsewhere. Pre-registered predictions: {sa}; {stt} → **{verdict}**.

  Standalone public repo with the instrument, plan, raw trials and results: https://github.com/kilojoules/llm-opinion-survival.

  ![orderforce](experiments/orderforce_figure.png)
{END}"""
    rows = []
    for a in POL + ["fair_month", "sign_color"]:
        x = AT[a]
        rows.append(f"| {a}{' (null)' if a in ('fair_month', 'sign_color') else ''} | {x['class']} | {x['n']} | {x['not called']} | {x['refusal-gated']} | {x['called']} | {x['+order']} | {x['+forcing']} | {x['survives all']} | {f2(x['d_ref_mean'])} | {x['revealed_called'] if x['revealed_called'] is not None else '—'} |")
    mrows = []
    for t in tags:
        M = R["models"][t]; s_ = M["spin"]; rs = [M["axes"][a]["rung"] for a in POL if a in M["axes"]]
        mrows.append(f"| {t} | {'yes' if not M['instrument_unreliable'] else 'no'} | {'yes' if M['numbering_limited'] else 'no'} | {f2(s_['order_F3'])} | {f2(s_['numbering'])} | {f2(s_['forcing'])} | {f2(s_['paraphrase_4'])} | {f2(s_['order_F3_pagree'])} | {f2(s_['d_order_range_mean'])} | "
                     f"{f2(M['position_index'])} | {f2(M['sweep_pos_effect'])} | {sum(r in ('called','+order','+forcing','survives all') for r in rs)} / {rs.count('+order') + rs.count('survives all')} / {rs.count('+forcing') + rs.count('survives all')} / {rs.count('survives all')} |")
    report = f"""{BEGIN}
## 13. Which stated preferences survive re-ordering the options and forcing the answer?

Röttger et al. vary the question wording and how hard the model is pushed; they hold the option list fixed. The
multiple-choice literature says the list itself is a large nuisance (models pick by position and by printed number), and
survey methodology says a legitimate "don't know" changes who answers. The pre-registered design (`experiments/orderforce/PLAN.md`)
crosses, inside the paper's own paradigm, four option orders (a Latin square closed under reversal) × two numberings
(positional vs pinned to the labels) at forcing 3, forcing levels 1 / 2 / 3 / 5 at the paper's order, and an explicit
"5) No opinion" at forcing 1 and 2 (last; crossed with the four orders; and listed first as a position control), on the 34
issue statements, the 62 PCT propositions, 4 null statements and 12 factual anchors, over the paper's 4 shared stems
(2 for the reduced arms) — 3,788 prompts per model, {len(tags)} models. A three-model pilot gated the launch (identity
reproduction against the 31-model run, greedy-decode agreement under permuted layouts, the opt-out substantive-mass pivot,
anchor comprehension); outcomes and amendments are appended to the plan.

The unit is d = (p_agree(report-level statement) − p_agree(opposing statement)) / 2, which cancels acquiescence and
content-blind position bias; raw item scores are reported beside it as the paper's quantity. Ladder per model × issue:
called at the reference cell (CI excludes 0, |d| ≥ {R['floor']}, BH q = 0.01 over 19 axes) → survives order + numbering (same
sign in all but one of the 7 cells, no significant dissent, every shift smaller than the reference signal) → survives
forcing + opt-out (same sign at forcing 5 with a shift smaller than the signal; same sign with the "No opinion" option
offered, across ≥ 3 of 4 orders) → survives all. Cells where the anchors show the model answering by number alone are
excluded per model as comprehension-broken (amendment).

| axis | class | models | not called | refusal-gated | called | +order | +forcing | survives all | mean d_ref | models with a called revealed lean |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

| model | reliable | numbering-limited | spin: order (Likert) | numbering | forcing | paraphrase (4) | order range of p_agree | order range of d | position index | conjoint listing-order effect | called / +order / +forcing / all |
|---|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(mrows) + f"""

Reading: {headline.strip('*')}. Order spin {f2(order)} Likert units vs paraphrase {f2(para)}, forcing {f2(forc)}, numbering {f2(numb)}; the paired d's
order range is {f2(d_rng)} against {f2(pa_rng)} for raw p_agree. Ladder over {len(rel)} reliable models: {called} of {n_cells} cells called, {surv_order} survive order and
numbering, {surv_forc} survive forcing and the opt-out, {counts['survives all']} survive all; {counts['refusal-gated']} refusal-gated; null axes called {R['null_called']} of {2 * len(rel)}.
Opt-out: median substantive mass {f2(min(dk_sub)) if dk_sub else '—'}–{f2(max(dk_sub)) if dk_sub else '—'} at forcing 2 with the out offered, uptake {f2(P.get('dk_rate_overall'))} overall. Join: ρ {f2(rho_all)} all / {f2(rho_s)}
surviving / {f2(rho_r)} rest (p = {f3(a6.get('p_one_sided'))}); top-tertile survival {f2(mt['revealed_called_survival'])} (revealed-called) vs {f2(mt['non_called_survival'])}. Predictions: {sa}; {stt} → {verdict}.
Caveats (PLAN §10–11): "survives" is invariance to these manipulations, not a true preference; the pair statements are
matched-strength policies, not negations; one opt-out wording; four paraphrases per cell; small models over-represented;
no cross-instrument magnitude claims.

![orderforce](../../orderforce_figure.png)

![ladder](orderforce_ladder.png)

![spin](orderforce_spin.png)
{END}"""
    if "--apply" in sys.argv:
        for path, text, anchor in ((ROOT / "README.md", readme, "<!-- esteer:end -->\n"),
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
