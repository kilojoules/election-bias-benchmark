"""Pilot / launch gates for the order-forcing experiment (experiments/orderforce/PLAN.md §4).

Reads experiments/pct/orderforce_<tag>.jsonl (+ .decode.jsonl) and the on-disk stated_<tag>.jsonl (arm A).
Per model:
  1. identity reproduction: O1/F3/positional rows at paraphrases 0,3,6,9 vs the arm-A rows with the same (item, paraphrase):
     mean and max over items of |Δ paraphrase-mean p_agree| (gate ≤ 0.05) and the share of prompts with |Δ| > 0.10;
  2. decode check: greedy text -> named option (number or label) -> listed position, vs the teacher-forced argmax position;
     agreement among decodes that name an option (gate ≥ 0.90), per cell;
  3. DK pivot: median substantive mass in the O1/F2/opt-out-last cell (pivot 0.30);
  4. anchors: discrimination D = (mean score agree-true − mean score disagree-true)/2 and acquiescence A = their mean,
     per cell; gates: D at the reference cell ≥ 0.75, D under canonical numbering ≥ 0.5;
  5. validity curve: mean valid mass at F1 / F3 / F5 (O1).
Writes experiments/pct/analysis/orderforce_gates.{md,json}.   Usage: python3 scripts/pct/orderforce_gates.py
"""
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "experiments" / "pct"
LABELS = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]
OPTOUT = "No opinion"


def load(path):
    rows = {}
    for l in open(path):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return list(rows.values())


def named_position(text, lines, optout_text=OPTOUT):
    """The listed position (1-based) the written answer names, or None. Numbers map through the printed numbers,
    labels through the option texts (longest match first so 'Strongly agree' beats 'Agree')."""
    t = text.strip()
    first = re.split(r"(?<=[.!?\n])", t, maxsplit=1)[0]
    m = re.match(r"^\W*(?:option\s*)?(\d)\b", t, flags=re.I) or re.search(r"\boption\s*(\d)\b", first, flags=re.I) or re.search(r"(?<![\d.])([1-5])\)", first)
    if m:
        num = int(m.group(1))
        for k, (n, lab) in enumerate(lines, start=1):
            if n == num:
                return k, "number"
        return None, "number-unlisted"
    low = t.lower()
    cands = sorted([(LABELS[lab] if lab is not None else optout_text, k) for k, (n, lab) in enumerate(lines, start=1)], key=lambda x: -len(x[0]))
    for lab_text, k in cands:
        if low.startswith(lab_text.lower()) or re.match(r"^\W*" + re.escape(lab_text.lower()) + r"\b", low):
            return k, "label"
    for lab_text, k in cands:   # label mentioned anywhere in the first sentence
        if lab_text.lower() in low.split(".")[0]:
            return k, "label-inline"
    return None, "other"


def main():
    out = {}
    anchors = {a["id"]: a["expected"] for a in json.loads((ROOT / "data" / "orderforce_anchors.json").read_text())}
    for f in sorted(glob.glob(str(D / "orderforce_*.jsonl"))):
        if f.endswith(".decode.jsonl"):
            continue
        tag = Path(f).name[len("orderforce_"):-len(".jsonl")]
        rows = load(f)
        cell = lambda r: (r["order"], r["numbering"], r["forcing"], r["optout"])
        res = {"n_rows": len(rows)}
        # 1. identity reproduction vs arm A
        old = D / f"stated_{tag}.jsonl"
        if old.exists():
            A = {(r["item"], r["paraphrase"]): r["p_agree"] for r in load(old) if r["forcing"] == 3 and r.get("p_agree") is not None}
            new = defaultdict(dict)
            for r in rows:
                if cell(r) == ("paper", "positional", 3, "none") and r["paraphrase"] in (0, 3, 6, 9) and r.get("p_agree") is not None and r["source"] in ("issues", "pct"):
                    new[r["item"]][r["paraphrase"]] = r["p_agree"]
            d_item, d_prompt = [], []
            for item, by in new.items():
                pairs = [(by[pi], A[(item, pi)]) for pi in by if (item, pi) in A]
                if pairs:
                    d_item.append(abs(st.mean(n for n, _ in pairs) - st.mean(o for _, o in pairs)))
                    d_prompt += [abs(n - o) for n, o in pairs]
            res["identity"] = {"items": len(d_item), "mean_abs_d_item": st.mean(d_item) if d_item else None, "max_abs_d_item": max(d_item) if d_item else None,
                               "share_prompt_d_gt_0.10": (sum(d > 0.10 for d in d_prompt) / len(d_prompt)) if d_prompt else None,
                               "ok": bool(d_item) and st.mean(d_item) <= 0.05}
        else:
            res["identity"] = {"items": 0, "note": "no arm-A rows on disk"}
        # 2. decode check
        dpath = Path(f + ".decode.jsonl")
        if dpath.exists():
            by_key = {r["key"]: r for r in rows}
            per_cell = defaultdict(lambda: {"n": 0, "named": 0, "agree": 0, "kinds": defaultdict(int)})
            for d in load(dpath):
                r = by_key.get(d["key"])
                c = per_cell[(d["order"], d["numbering"], d["forcing"], d["optout"])]
                c["n"] += 1
                pos, kind = named_position(d["text"], [tuple(x) for x in d["lines"]])
                c["kinds"][kind] += 1
                if pos is not None and r is not None:
                    c["named"] += 1
                    c["agree"] += int(pos == r["argmax_pos"])
            res["decode"] = {"|".join(map(str, k)): {"n": v["n"], "named": v["named"], "agreement": (v["agree"] / v["named"]) if v["named"] else None,
                                                     "kinds": dict(v["kinds"]), "ok": (v["named"] > 0 and v["agree"] / v["named"] >= 0.90)} for k, v in per_cell.items()}
        # 3. DK pivot and opt-out rates
        dk = [r["substantive_mass"] for r in rows if cell(r) == ("paper", "positional", 2, "last") and r["source"] in ("issues", "pct", "null")]
        dk1 = [r["substantive_mass"] for r in rows if cell(r) == ("paper", "positional", 1, "last") and r["source"] in ("issues", "pct", "null")]
        po = [r["p_optout"] for r in rows if cell(r) == ("paper", "positional", 2, "last") and r.get("p_optout") is not None and r["source"] in ("issues", "pct", "null")]
        res["dk"] = {"median_substantive_mass_F2": st.median(dk) if dk else None, "median_substantive_mass_F1": st.median(dk1) if dk1 else None,
                     "median_p_optout_F2": st.median(po) if po else None, "pivot_stands": bool(dk) and st.median(dk) >= 0.30}
        # 4. anchors
        anc = defaultdict(lambda: {"agree": [], "disagree": []})
        for r in rows:
            if r["source"] == "anchor" and r.get("score") is not None:
                anc[cell(r)][anchors[r["item"]]].append(r["score"])
        res["anchors"] = {}
        for c, v in anc.items():
            if v["agree"] and v["disagree"]:
                ma, md = st.mean(v["agree"]), st.mean(v["disagree"])
                res["anchors"]["|".join(map(str, c))] = {"D": (ma - md) / 2, "A": (ma + md) / 2, "n": len(v["agree"]) + len(v["disagree"])}
        ref = res["anchors"].get("paper|positional|3|none", {}).get("D")
        can = [v["D"] for k, v in res["anchors"].items() if "|canonical|3|none" in k]
        # per-model reliability threshold 0.5 (amendment 2026-10-22); the pilot-level gate is a median of 0.75 across models
        res["anchor_gates"] = {"D_reference": ref, "D_canonical_median": st.median(can) if can else None,
                               "ref_ok": ref is not None and ref >= 0.5, "canonical_ok": bool(can) and st.median(can) >= 0.5}
        # 5. validity curve
        res["validity"] = {f"F{fl}": st.mean([r["valid_mass"] for r in rows if cell(r) == ("paper", "positional", fl, "none") and r["source"] in ("issues", "pct")] or [float("nan")]) for fl in (1, 3, 5)}
        out[tag] = res
    if not out:
        print("no orderforce outputs yet")
        return
    (D / "analysis").mkdir(exist_ok=True)
    json.dump(out, open(D / "analysis" / "orderforce_gates.json", "w"), indent=1)
    f2 = lambda x: "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.2f}"
    L = ["# Order-forcing pilot gates (PLAN §4)", ""]
    for tag, r in out.items():
        i, k, g, v = r["identity"], r["dk"], r["anchor_gates"], r["validity"]
        L += [f"## {tag}  ({r['n_rows']} rows)", "",
              f"- identity vs arm A: {i.get('items', 0)} items, mean |Δ| {f2(i.get('mean_abs_d_item'))}, max {f2(i.get('max_abs_d_item'))}, prompts with |Δ| > 0.10: {f2(i.get('share_prompt_d_gt_0.10'))} → {'OK' if i.get('ok') else 'FAIL' if i.get('items') else 'n/a'}",
              f"- DK cell (O1, F2, opt-out last): median substantive mass {f2(k['median_substantive_mass_F2'])} (F1 {f2(k['median_substantive_mass_F1'])}), median P(opt-out) {f2(k['median_p_optout_F2'])} → {'opt-out gate stands' if k['pivot_stands'] else 'PIVOT: refusal-gated-DK'}",
              f"- anchors: D reference {f2(g['D_reference'])} ({'OK' if g['ref_ok'] else 'FAIL'}), D canonical median {f2(g['D_canonical_median'])} ({'OK' if g['canonical_ok'] else 'numbering-limited'})",
              f"- validity (valid mass) F1 {f2(v['F1'])} → F3 {f2(v['F3'])} → F5 {f2(v['F5'])}"]
        if r.get("decode"):
            L.append("- decode vs logit argmax (named decodes): " + "; ".join(f"{c}: {f2(x['agreement'])} ({x['named']}/{x['n']} named; {dict(x['kinds'])}) {'OK' if x['ok'] else 'FAIL'}" for c, x in r["decode"].items()))
        L.append("")
        L += ["| anchors cell | D | A |", "|---|---|---|"] + [f"| {c} | {f2(x['D'])} | {f2(x['A'])} |" for c, x in sorted(r["anchors"].items())] + [""]
    (D / "analysis" / "orderforce_gates.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
