"""Summarise the decode-vs-logit check (scripts/pct/decode_check.py) across models.

Reads experiments/pct/decode_<tag>.jsonl and writes experiments/pct/analysis/decode_check.md: per model the
fraction of trials whose text names a candidate, the agreement between the text's pick and the logit
argmax, the listing-position effect in text vs logits, per-issue agreement, and the behaviour of the
three models excluded by naming mass (Phi-4, Gemma 4 12B, Gemma 4 26B-A4B are not in this run; the
check covers the models that were).

Usage: python3 scripts/pct/decode_analyze.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ROSTER = json.loads((ROOT / "scripts" / "sweep_models.json").read_text())


def main():
    sys.path.insert(0, str(ROOT / "scripts" / "pct"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from decode_check import pick_from_text  # noqa: E402  (lazy: imports torch)
    d = ROOT / "experiments" / "pct"
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fam = {m["tag"]: (m.get("family"), m.get("params_b")) for m in ROSTER["models"]}
    L = ["# Decode-vs-logit check", "",
         "For 512 conjoint trials per model (8 flipped issues, property-tax focal, 8 backgrounds, both orders, 4 name draws) the model's "
         "answer was greedily decoded (48 tokens) and the candidate it names first compared with the argmax of the teacher-forced "
         "name log-probs that the sweep reads. 'named' = fraction of trials whose text names either candidate; 'agreement' = "
         "P(text pick == logit pick) on those trials; position = P(first-listed candidate is picked), both columns over the named trials. "
         "A pick requires the answer's first paragraph to name exactly one candidate; 'Both ...'/'Neither ...' or naming both is ambiguous. "
         "Picks are re-parsed from the stored text (first 200 characters); 'path' = scoring path chosen by the per-run self-check.", "",
         "| model | family | params (B) | path | n | named | agreement | text picks first | logit picks first | none/ambiguous | example answer |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    rows_all = []
    for f in sorted(glob.glob(str(d / "decode_*.jsonl"))):
        tag = Path(f).name[len("decode_"):-len(".jsonl")]
        rows = []
        for l in open(f):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass
        if not rows:
            continue
        for r in rows:   # re-parse the stored text with the current (conservative) rules so every model is treated alike
            r["decoded_pick"] = pick_from_text(r["text"], r["names"][0], r["names"][1])
            r["agree"] = (r["decoded_pick"] == r["logit_pick"]) if r["decoded_pick"] in ("first", "second") else None
        named = [r for r in rows if r["decoded_pick"] in ("first", "second")]
        agree = sum(1 for r in named if r["agree"]) / max(len(named), 1)
        tf = sum(1 for r in named if r["decoded_pick"] == "first") / max(len(named), 1)
        lf = sum(1 for r in named if r["logit_pick"] == "first") / max(len(named), 1)
        nonef = 1 - len(named) / len(rows)
        ex = next((r["text"].replace("\n", " ")[:60] for r in rows if r["decoded_pick"] not in ("first", "second")), rows[0]["text"].replace("\n", " ")[:60])
        summary[tag] = {"n": len(rows), "path": rows[0].get("path"), "named": len(named) / len(rows), "agreement": agree, "text_first": tf, "logit_first": lf}
        for r in rows:
            r["tag"] = tag
        rows_all += rows
        f_, p_ = fam.get(tag, (None, None))
        path = rows[0].get("path", "cache?")
        L.append(f"| {tag} | {f_ or ''} | {p_ or ''} | {path} | {len(rows)} | {len(named)/len(rows):.2f} | {agree:.3f} | {tf:.2f} | {lf:.2f} | {nonef:.2f} | {ex} |")
    if summary:
        vals = sorted(s["agreement"] for s in summary.values())
        nm = sorted(s["named"] for s in summary.values())
        L += ["", f"{len(summary)} models. Agreement: median {vals[len(vals)//2]:.3f}, min {vals[0]:.3f}, max {vals[-1]:.3f}. "
              f"Named fraction: median {nm[len(nm)//2]:.2f}, min {nm[0]:.2f}. "
              f"Models with agreement < 0.9: {[t for t, s in summary.items() if s['agreement'] < 0.9]}. "
              f"Models whose text names a candidate in < 50% of trials: {[t for t, s in summary.items() if s['named'] < 0.5]}."]
        # per-issue agreement and per-issue win-rate from text vs logits
        by = defaultdict(lambda: {"n": 0, "agree": 0, "text_report": 0, "logit_report": 0, "named": 0})
        for r in rows_all:
            b = by[r["flip"]]
            b["n"] += 1
            if r["decoded_pick"] in ("first", "second"):
                b["named"] += 1
                b["agree"] += bool(r["agree"])
                b["text_report"] += (r["decoded_pick"] == "first") == r["report_first"]
            b["logit_report"] += (r["logit_pick"] == "first") == r["report_first"]
        L += ["", "## Per issue (all models pooled)", "", "| issue | named | agreement | P(report level) from text | from logits |", "|---|---|---|---|---|"]
        for a, b in by.items():
            L.append(f"| {a} | {b['named']/b['n']:.2f} | {b['agree']/max(b['named'],1):.3f} | {b['text_report']/max(b['named'],1):.3f} | {b['logit_report']/b['n']:.3f} |")
    (out / "decode_check.md").write_text("\n".join(L) + "\n")
    json.dump(summary, open(out / "decode_check.json", "w"), indent=1)
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
