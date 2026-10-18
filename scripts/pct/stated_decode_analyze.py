"""Compare the decoded stated-opinion readout (scripts/pct/stated_decode.py) with the logit one.

Per model and forcing level: the share of decoded answers that name an option, refuse, or do
neither; the mean answer-label mass (valid_mass from stated_<tag>.jsonl) within each category;
and, where the text names an option, how often it is the logit argmax. Writes
experiments/pct/analysis/stated_decode.md / .json.

Usage: python3 scripts/pct/stated_decode_analyze.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "pct"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    from stated_decode import classify  # noqa: E402  (lazy: importing torch is slow)
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    L = ["# Decoded stated readout: is low answer-label mass a refusal?", "",
         "The paper's prompt (3 paraphrases x forcing levels 1 and 5 x 96 items) greedily decoded and classified: *answer* = one of the "
         "four options named before any refusal phrase; *refusal* = \"as an AI / I don't have personal opinions\" wording; *other* = "
         "neither. valid mass = probability on the four answer labels from the logit readout on the same prompt "
         "(`stated_<tag>.jsonl`). 'option agreement' = share of decoded answers whose option is the logit argmax.", "",
         "| model | forcing | n | answer | refusal | other | valid mass: answers | valid mass: refusals | option agreement |",
         "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    pooled = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(d / "sdecode_*.jsonl"))):
        tag = Path(f).name[len("sdecode_"):-len(".jsonl")]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        for r in rows:   # classify from the stored text so every model uses the same (current) rules
            r["category"], r["option"] = classify(r["text"])
        logit = {}
        lf = d / f"stated_{tag}.jsonl"
        if lf.exists():
            for l in open(lf):
                try:
                    r = json.loads(l)
                    logit[r["key"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        summary[tag] = {}
        for fl in sorted({r["forcing"] for r in rows}):
            rs = [r for r in rows if r["forcing"] == fl]
            n = len(rs)
            cats = {c: sum(1 for r in rs if r["category"] == c) / n for c in ("answer", "refusal", "other")}
            vm = {c: [logit[r["key"]]["valid_mass"] for r in rs if r["category"] == c and r["key"] in logit and logit[r["key"]].get("valid_mass") is not None]
                  for c in ("answer", "refusal")}
            agree = [int(r["option"] - 1 == max(range(4), key=lambda k: logit[r["key"]]["lp_options"][k]))
                     for r in rs if r["category"] == "answer" and r["key"] in logit and logit[r["key"]].get("lp_options")]
            mean = lambda xs: sum(xs) / len(xs) if xs else None
            summary[tag][fl] = {"n": n, **cats, "vm_answer": mean(vm["answer"]), "vm_refusal": mean(vm["refusal"]),
                                "option_agreement": mean(agree), "n_agree": len(agree)}
            for c in cats:
                pooled[fl][c].append(cats[c])
            fmt = lambda x: "—" if x is None else f"{x:.2f}"
            L.append(f"| {tag} | {fl} | {n} | {cats['answer']:.2f} | {cats['refusal']:.2f} | {cats['other']:.2f} | "
                     f"{fmt(summary[tag][fl]['vm_answer'])} | {fmt(summary[tag][fl]['vm_refusal'])} | {fmt(summary[tag][fl]['option_agreement'])} ({len(agree)}) |")
    if summary:
        L.append("")
        for fl in sorted(pooled):
            med = lambda xs: sorted(xs)[len(xs) // 2]
            L.append(f"Forcing {fl}, {len(pooled[fl]['answer'])} models: median decoded answer rate {med(pooled[fl]['answer']):.2f}, "
                     f"median refusal rate {med(pooled[fl]['refusal']):.2f}, median other {med(pooled[fl]['other']):.2f}.")
        va = [v[fl]["vm_answer"] for v in summary.values() for fl in v if v[fl]["vm_answer"] is not None]
        vr = [v[fl]["vm_refusal"] for v in summary.values() for fl in v if v[fl]["vm_refusal"] is not None]
        ag = [v[fl]["option_agreement"] for v in summary.values() for fl in v if v[fl]["option_agreement"] is not None and v[fl]["n_agree"] >= 20]
        if va and vr:
            L.append(f"Across model x forcing cells, mean valid mass is {sum(va)/len(va):.2f} on prompts the model answers in text and "
                     f"{sum(vr)/len(vr):.2f} on prompts it refuses in text" + (f"; where the text names an option it is the logit argmax {sum(ag)/len(ag):.2f} of the time (median over cells with >= 20 answers)." if ag else "."))
    (out / "stated_decode.md").write_text("\n".join(L) + "\n")
    json.dump(summary, open(out / "stated_decode.json", "w"), indent=1)
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
